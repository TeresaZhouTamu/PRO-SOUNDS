import argparse
import pandas as pd
import torch
import os
import numpy as np
import torch.nn.functional as F
from utils.esmc_utils import get_esmc_layer_and_feature_dim, load_esmc_model, extract_esmc_features

def compute_qc_weights(scores, uncertainties, target_high=True, penalty=1.0, contrast=2.0):
    """
    Computes weights based on Quality (Probability) and Confidence (Uncertainty).
    Uses 'Lower Confidence Bound' to penalize risky predictions.
    """
    s = np.array(scores, dtype=np.float32)
    u = np.array(uncertainties, dtype=np.float32)
    u = np.nan_to_num(u, nan=1.0) 

    if target_high:
        raw_signal = s
    else:
        raw_signal = 1.0 - s

    # Effective Signal = Signal - (Risk)
    effective_signal = raw_signal - (penalty * u)

    # If Risk > Signal, the sample is unreliable. Weight = 0.
    effective_signal = np.maximum(effective_signal, 0.0)

    # Sharpen with contrast
    weights = effective_signal ** contrast

    weight_sum = np.sum(weights)
    if weight_sum == 0:
        # Fallback if LCB filters everything out
        return np.ones_like(weights) / len(weights)
        
    return weights / weight_sum

def compute_weighted_mean(features, weights=None):
    if weights is None:
        return features.mean(dim=1)
        
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, device=features.device, dtype=features.dtype)
    else:
        weights = weights.to(features.device).to(features.dtype)
    
    # Reshape: (1, num_samples, 1)
    w_reshaped = weights.view(1, -1, 1)
    w_sum = w_reshaped.sum(dim=1, keepdim=True)
    w_normalized = w_reshaped / (w_sum + 1e-8)
    
    return (features * w_normalized).sum(dim=1)

def orthogonalize_vector(v_target, v_axis):
    """Projects v_target to be perpendicular to v_axis."""
    v_axis_norm = F.normalize(v_axis, p=2, dim=-1)
    projection = (v_target * v_axis_norm).sum(dim=-1, keepdim=True) * v_axis_norm
    return v_target - projection


def process_dataset(data_path, model, tokenizer, n_layers, device, batch_size, 
                   uncertainty_path=None, unc_col=None, score_col=None, lcb_penalty=1.0):
    """
    Loads data and extracts vectors.
    """
    print(f"--- Processing {data_path} ---")
    
    # Load Data
    df = pd.read_csv(data_path)
    
    use_weights = False
    
    if uncertainty_path and unc_col and score_col:
        print(f"  >>> Merging Uncertainty ('{unc_col}') and Score ('{score_col}')...")
        df_unc = pd.read_csv(uncertainty_path)
        
        # Merge
        df = pd.merge(df, df_unc[['sequence', unc_col, score_col]], on='sequence', how='inner')
        use_weights = True
    else:
        print("  >>> No uncertainty provided. Using standard mean (Trusted ID Data).")

    # Split Groups
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    
    if len(pos_df) == 0 or len(neg_df) == 0:
        raise ValueError("One of the groups is empty")

    # Extract Features
    print("  >>> Extracting features...")
    # Global family mean (unweighted)
    all_seqs = df['sequence'].tolist()
    all_feats = extract_esmc_features(all_seqs, model, tokenizer, n_layers, batch_size=batch_size, device=device)
    
    pos_feats = extract_esmc_features(pos_df['sequence'].tolist(), model, tokenizer, n_layers, batch_size=batch_size, device=device)
    neg_feats = extract_esmc_features(neg_df['sequence'].tolist(), model, tokenizer, n_layers, batch_size=batch_size, device=device)
    
    mu_global = all_feats.mean(dim=1)
    
    # Class Means
    if use_weights:
        print(f"  >>> Applying Quality-Confidence Weighting (Penalty={lcb_penalty})...")
        
        w_pos = compute_qc_weights(
            pos_df[score_col].values, pos_df[unc_col].values, 
            target_high=True, penalty=lcb_penalty
        )
        w_neg = compute_qc_weights(
            neg_df[score_col].values, neg_df[unc_col].values, 
            target_high=False, penalty=lcb_penalty
        )
        
        mu_pos = compute_weighted_mean(pos_feats, w_pos)
        mu_neg = compute_weighted_mean(neg_feats, w_neg)
    else:
        mu_pos = compute_weighted_mean(pos_feats, weights=None)
        mu_neg = compute_weighted_mean(neg_feats, weights=None)
    
    return mu_global, mu_pos - mu_neg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # source domain
    parser.add_argument('--id_data', type=str, required=True, help="ID dataset path")
    
    # target domain
    parser.add_argument('--ood_data', type=str, required=True, help="OOD dataset path")
    parser.add_argument('--ood_unc', type=str, required=True, help="OOD uncertainty & scores CSV")

    parser.add_argument('--uncertainty_type', type=str, default='epistemic', choices=['total', 'aleatoric', 'epistemic'])
    parser.add_argument('--score_col', type=str, default='pred_prob', help="Probability column name")
    
    parser.add_argument('--lcb_penalty', type=float, default=1.0, help="Strength of uncertainty penalty.")
    parser.add_argument('--save_folder', type=str, default="hybrid_qc_vectors")
    parser.add_argument('--property', type=str, default="immunog")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    col_map = {'total': 'total_uncertainty', 'aleatoric': 'aleatoric_uncertainty', 'epistemic': 'epistemic_uncertainty'}
    selected_unc_col = col_map[args.uncertainty_type]

    n_layers, dim = get_esmc_layer_and_feature_dim()
    model, tokenizer = load_esmc_model(device=args.device)

    print("\n=== Processing ID Data (Source) ===")
    mu_id_global, v_id = process_dataset(
        args.id_data, model, tokenizer, n_layers, args.device, args.batch_size,
        uncertainty_path=None 
    )

    print("\n=== Processing OOD Data (Target) ===")
    mu_ood_global, v_ood = process_dataset(
        args.ood_data, model, tokenizer, n_layers, args.device, args.batch_size,
        uncertainty_path=args.ood_unc, 
        unc_col=selected_unc_col, 
        score_col=args.score_col,
        lcb_penalty=args.lcb_penalty
    )

    v_id = v_id.to(args.device)
    v_ood = v_ood.to(args.device)
    mu_id_global = mu_id_global.to(args.device)
    mu_ood_global = mu_ood_global.to(args.device)

    print("\n=== Constructing Smart Hybrid Vector ===")

    v_fam = mu_ood_global - mu_id_global
    v_id_safe = orthogonalize_vector(v_id, v_fam)
    mag_id = v_id_safe.norm(p=2, dim=-1, keepdim=True)
    mag_ood = v_ood.norm(p=2, dim=-1, keepdim=True)
    
    v_id_norm = F.normalize(v_id_safe, p=2, dim=-1)
    v_ood_norm = F.normalize(v_ood, p=2, dim=-1)

    # Conflict Gating
    similarities = F.cosine_similarity(v_id_norm, v_ood_norm, dim=-1).unsqueeze(-1)
    ramp = torch.linspace(0.9, 0.2, steps=n_layers, device=args.device).unsqueeze(-1)

    # print("Layer-wise Agreements (Cosine Sim):")
    # for i in range(0, similarities.shape[0], 5): 
    #     print(f"  Layer {i}: {similarities[i].item():.3f}")
    
    conflict_threshold = -0.05
    conflict_mask = (similarities < conflict_threshold).float()
    final_alpha = (ramp * (1 - conflict_mask)) + (1.0 * conflict_mask)

    # print(f"\nConstructing Hybrid... Avg Alpha: {final_alpha.mean().item():.3f}")
    # print(f"Conflict Gating triggered on {conflict_mask.sum().item()} layers (Forced to ID vector).")
    
    v_hybrid_dir = (final_alpha * v_id_norm) + ((1 - final_alpha) * v_ood_norm)
    v_hybrid_dir = F.normalize(v_hybrid_dir, p=2, dim=-1)
    final_mag = torch.max(mag_id, mag_ood)
    v_hybrid = v_hybrid_dir * final_mag

    os.makedirs(args.save_folder, exist_ok=True)
    save_path = f"{args.save_folder}/Hybrid_{args.property}_QC_Penalty{args.lcb_penalty}.pt"
    dummy_neg = torch.zeros_like(v_hybrid).cpu()
    torch.save((v_hybrid.cpu(), dummy_neg), save_path)
    print(f"\nSaved Hybrid Vector (Compatibility Mode) to: {save_path}")