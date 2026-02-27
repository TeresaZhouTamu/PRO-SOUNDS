import argparse
import pandas as pd
import torch
import os
import numpy as np
import torch.nn.functional as F
from utils.esmc_utils import get_esmc_layer_and_feature_dim, load_esmc_model, extract_esmc_features

def compute_robust_weights(uncertainties, clip_percentile=10):
    """Computes robust normalized inverse-variance weights."""
    vals = np.array(uncertainties, dtype=np.float32)
    vals = np.nan_to_num(vals, nan=1.0) 
    limit = np.percentile(vals, clip_percentile)
    if limit == 0: limit = 1e-6
    vals_clipped = np.maximum(vals, limit)
    raw_weights = 1.0 / (vals_clipped ** 2)
    normalized_weights = raw_weights / np.sum(raw_weights)
    return normalized_weights

def compute_mean(features, weights=None):
    if weights is None:
        return features.mean(dim=1)
        
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, device=features.device, dtype=features.dtype)
    else:
        weights = weights.to(features.device).to(features.dtype)
        
    w_reshaped = weights.view(1, -1, 1)
    w_sum = w_reshaped.sum(dim=1, keepdim=True)
    w_normalized = w_reshaped / (w_sum + 1e-8)

    return (features * w_normalized).sum(dim=1)

def orthogonalize_vector(v_target, v_axis):
    """Projects v_target to be perpendicular (orthogonal) to v_axis."""
    v_axis_norm = F.normalize(v_axis, p=2, dim=-1)
    projection_scalar = (v_target * v_axis_norm).sum(dim=-1, keepdim=True)
    v_parallel = projection_scalar * v_axis_norm
    v_ortho = v_target - v_parallel
    return v_ortho

def process_dataset(data_path, model, tokenizer, n_layers, device, batch_size, 
                   uncertainty_path=None, col_name=None, limit=None):
    print(f"--- Processing {data_path} ---")
    df = pd.read_csv(data_path)
    
    use_weights = False
    if uncertainty_path and col_name:
        print(f"  >>> Loading uncertainty from {uncertainty_path}...")
        df_unc = pd.read_csv(uncertainty_path)
        if col_name not in df_unc.columns:
            raise KeyError(f"Column '{col_name}' not found in {uncertainty_path}")
        
        df = pd.merge(df, df_unc[['sequence', col_name]], on='sequence', how='inner')
        use_weights = True
    else:
        print("  >>> No uncertainty provided. Using standard mean (Ground Truth labels assumed).")

    if limit: df = df.head(limit)
    
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    
    print(f"  >>> Found {len(pos_df)} Pos and {len(neg_df)} Neg samples.")
    if len(pos_df) == 0 or len(neg_df) == 0:
        raise ValueError("One of the groups is empty! Check labels.")

    # ALL features for the global centroids
    all_seqs = df['sequence'].tolist()
    pos_seqs = pos_df['sequence'].tolist()
    neg_seqs = neg_df['sequence'].tolist()
    
    print("  >>> Extracting features...")
    all_feats = extract_esmc_features(all_seqs, model, tokenizer, n_layers, batch_size=batch_size, device=device)
    pos_feats = extract_esmc_features(pos_seqs, model, tokenizer, n_layers, batch_size=batch_size, device=device)
    neg_feats = extract_esmc_features(neg_seqs, model, tokenizer, n_layers, batch_size=batch_size, device=device)
    
    # Center of the cluster
    global_mean = all_feats.mean(dim=1) 
    
    # Class Means (Weighted or Unweighted)
    if use_weights:
        # print("  >>> Applying uncertainty weighting...")
        pos_weights = compute_robust_weights(pos_df[col_name].values)
        neg_weights = compute_robust_weights(neg_df[col_name].values)
        mu_pos = compute_mean(pos_feats, pos_weights)
        mu_neg = compute_mean(neg_feats, neg_weights)
    else:
        mu_pos = compute_mean(pos_feats, weights=None)
        mu_neg = compute_mean(neg_feats, weights=None)
    
    # Steering Vector (Pos - Neg)
    steering_vec = mu_pos - mu_neg
    
    return global_mean, steering_vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id_data', type=str, required=True, help="ID dataset (sequences + labels)")
    
    parser.add_argument('--ood_data', type=str, required=True, help="OOD dataset (sequences + labels)")
    parser.add_argument('--ood_unc', type=str, required=True, help="OOD uncertainty scores")
    parser.add_argument('--uncertainty_type', type=str, default='epistemic', choices=['total', 'aleatoric', 'epistemic'])
    
    parser.add_argument('--alpha', type=float, default=0.7, help="Mixing weight. 1.0 = Pure Safe ID, 0.0 = Pure OOD.")
    parser.add_argument('--save_folder', type=str, default="hybrid_vectors")
    parser.add_argument('--property', type=str, default="immunog")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    col_map = {
        'total': 'total_uncertainty', 
        'aleatoric': 'aleatoric_uncertainty', 
        'epistemic': 'epistemic_uncertainty'
    }
    selected_col = col_map[args.uncertainty_type]

    n_layers, feature_dim = get_esmc_layer_and_feature_dim()
    try:
        model, tokenizer = load_esmc_model(device=args.device)
    except Exception as e:
        print(f"Error loading ESMC model: {e}")
        exit(1)

    # ID Labeled Data --> Standard Mean
    mu_id_global, v_id = process_dataset(
        data_path=args.id_data, 
        model=model, tokenizer=tokenizer, n_layers=n_layers, device=args.device, batch_size=args.batch_size,
        uncertainty_path=None, col_name=None 
    )

    # OOD UN-labeled Data -> Weighted Mean
    mu_ood_global, v_ood = process_dataset(
        data_path=args.ood_data, 
        model=model, tokenizer=tokenizer, n_layers=n_layers, device=args.device, batch_size=args.batch_size,
        uncertainty_path=args.ood_unc, col_name=selected_col
    )

    print("\n--- Computing Hybrid Vector ---")
    v_id_dev = v_id.to(args.device)
    v_ood_dev = v_ood.to(args.device)
    mu_ood_global_dev = mu_ood_global.to(args.device)
    mu_id_global_dev = mu_id_global.to(args.device)

    # Orthogonalize ID as ref vectors
    v_fam = mu_ood_global_dev - mu_id_global_dev
    v_id_safe = orthogonalize_vector(v_id_dev, v_fam)

    # Normalize
    mag_id = v_id_safe.norm(p=2, dim=-1, keepdim=True)
    mag_ood = v_ood_dev.norm(p=2, dim=-1, keepdim=True)

    v_id_norm = F.normalize(v_id_safe, p=2, dim=-1)
    v_ood_norm = F.normalize(v_ood_dev, p=2, dim=-1)

    # Layer-Wise Cosine Similarity
    similarities = F.cosine_similarity(v_id_norm, v_ood_norm, dim=-1).unsqueeze(-1)
    
    # print("Layer-wise Agreements (Cosine Sim):")
    # for i in range(0, similarities.shape[0], 5): 
    #     print(f"  Layer {i}: {similarities[i].item():.3f}")

    # Layers 0-15: mostly ID (Structural integrity)
    # Layers 16-36: mostly OOD (Functional specificity)
    num_layers = v_id.shape[0]
    start_alpha = 1.0
    end_alpha = 0.1
    ramp = torch.linspace(start_alpha, end_alpha, steps=num_layers, device=args.device).unsqueeze(-1)
    
    # IF conflict -> use ID solely; otherwise use HYBRID
    conflict_threshold = -0.05
    conflict_mask = (similarities < conflict_threshold).float()
    final_alpha = (ramp * (1 - conflict_mask)) + (1.0 * conflict_mask)

    print(f"\nConstructing Hybrid... Avg Alpha: {final_alpha.mean().item():.3f}")
    print(f"Conflict Gating triggered on {conflict_mask.sum().item()} layers (Forced to ID vector).")

    # v_hybrid = alpha * v_ID + (1-alpha) * v_OOD
    v_hybrid_dir = (final_alpha * v_id_norm) + ((1 - final_alpha) * v_ood_norm)

    v_hybrid_dir = F.normalize(v_hybrid_dir, p=2, dim=-1)
    
    final_mag = torch.max(mag_id, mag_ood)
    v_hybrid = v_hybrid_dir * final_mag

    os.makedirs(args.save_folder, exist_ok=True)
    save_path = f"{args.save_folder}/Hybrid_{args.property}_Gated.pt"
    
    torch.save({
        'hybrid_vector': v_hybrid.cpu(),
        'v_id_safe': v_id_safe.cpu(),
        'v_ood': v_ood_dev.cpu(),
        'final_alpha': final_alpha.cpu(),
        'conflict_mask': conflict_mask.cpu()
    }, save_path)
    print(f"Hybrid vector saved to: {save_path}")