import argparse
import pandas as pd
import torch
import os
import numpy as np
import sys
from utils.esmc_utils import get_esmc_layer_and_feature_dim, load_esmc_model, extract_esmc_features

def compute_quality_confidence_weights(scores, uncertainties, target_high=True, 
                                     penalty_std=1.0, contrast=2.0):
    """
    Computes weights based on the 'Safe Signal' (Lower Confidence Bound).
    """
    s = np.array(scores, dtype=np.float32)
    u = np.array(uncertainties, dtype=np.float32)
    u = np.nan_to_num(u, nan=1.0) 

    if target_high:
        raw_signal = s
    else:
        raw_signal = 1.0 - s

    # Effective Signal = Raw Signal - (Risk Factor)
    effective_signal = raw_signal - (penalty_std * u)

    # If the Risk > Signal, the effective signal is < 0. We clip it to 0.
    effective_signal = np.maximum(effective_signal, 0.0)
    weights = effective_signal ** contrast
    weight_sum = np.sum(weights)
    
    if weight_sum == 0:
        print(">>> Warning: All samples were filtered out by LCB. Back to simple inverse-variance.")
        fallback = 1.0 / (u**2 + 1e-6)
        return fallback / fallback.sum()
        
    return weights / weight_sum

def compute_weighted_mean(features, weights):
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, device=features.device, dtype=features.dtype)
    else:
        weights = weights.to(features.device).to(features.dtype)
        
    w_reshaped = weights.view(1, -1, 1)
    w_sum = w_reshaped.sum(dim=1, keepdim=True) + 1e-8
    return (features * w_reshaped / w_sum).sum(dim=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate QC-Weighted Steering Vectors")
    
    # Inputs
    parser.add_argument('--data_path', type=str, required=True, help="Path to CSV with 'sequence', 'label'.")
    parser.add_argument('--uncertainty_path', type=str, required=True, help="Path to CSV with scores & uncertainty.")
    
    # Columns
    parser.add_argument('--uncertainty_type', type=str, default='epistemic', 
                        choices=['total', 'aleatoric', 'epistemic'], help="Which uncertainty col to use.")
    parser.add_argument('--score_col', type=str, default='pred_prob', 
                        help="Column containing the surrogate model probability score (0-1).")

    parser.add_argument('--lcb_penalty', type=float, default=1.0, 
                        help="Strength of uncertainty penalty. 1.0 = Standard LCB. Higher filters more aggressively.")
    parser.add_argument('--property', type=str, required=True)
    parser.add_argument('--save_folder', type=str, default="weighted_vectors")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    unc_map = {
        'total': 'total_uncertainty',
        'aleatoric': 'aleatoric_uncertainty',
        'epistemic': 'epistemic_uncertainty'
    }
    selected_unc_col = unc_map.get(args.uncertainty_type, args.uncertainty_type)

    print(f"Loading data from {args.data_path}...")
    df_data = pd.read_csv(args.data_path)
    df_unc = pd.read_csv(args.uncertainty_path)
    
    if args.score_col not in df_unc.columns:
        raise KeyError(f"Score column '{args.score_col}' not found in uncertainty file.")
    merged_df = pd.merge(df_data, df_unc[['sequence', selected_unc_col, args.score_col]], on='sequence', how='inner')
    
    pos_df = merged_df[merged_df['label'] == 1].copy()
    neg_df = merged_df[merged_df['label'] == 0].copy()
    
    if len(pos_df) == 0 or len(neg_df) == 0:
        raise ValueError("One of the groups is empty!")

    print(f"Processing {len(pos_df)} Positive and {len(neg_df)} Negative sequences.")

    pos_weights_np = compute_quality_confidence_weights(
        scores=pos_df[args.score_col].values,
        uncertainties=pos_df[selected_unc_col].values,
        target_high=True, # We want High Scores
        penalty_std=args.lcb_penalty
    )

    # Signal = (1.0 - Score)
    neg_weights_np = compute_quality_confidence_weights(
        scores=neg_df[args.score_col].values,
        uncertainties=neg_df[selected_unc_col].values,
        target_high=False, # We want Low Scores
        penalty_std=args.lcb_penalty
    )
    
    # best_pos_idx = np.argmax(pos_weights_np)
    # best_neg_idx = np.argmax(neg_weights_np)
    
    # print(f"  [Pos] Best Sample: Score={pos_df[args.score_col].values[best_pos_idx]:.4f}, Unc={pos_df[selected_unc_col].values[best_pos_idx]:.4f}")
    # print(f"  [Neg] Best Sample: Score={neg_df[args.score_col].values[best_neg_idx]:.4f}, Unc={neg_df[selected_unc_col].values[best_neg_idx]:.4f}")

    # Extract Features
    n_layers, dim = get_esmc_layer_and_feature_dim()
    model, tokenizer = load_esmc_model(device=args.device)

    print("\nExtracting features...")
    pos_feats = extract_esmc_features(pos_df['sequence'].tolist(), model, tokenizer, n_layers, batch_size=args.batch_size, device=args.device)
    neg_feats = extract_esmc_features(neg_df['sequence'].tolist(), model, tokenizer, n_layers, batch_size=args.batch_size, device=args.device)

    print("Computing weighted vectors...")
    pos_vec = compute_weighted_mean(pos_feats, pos_weights_np)
    neg_vec = compute_weighted_mean(neg_feats, neg_weights_np)
    
    os.makedirs(args.save_folder, exist_ok=True)
    save_path = f"{args.save_folder}/Steering_{args.property}_QC_Penalty{args.lcb_penalty}.pt"
    torch.save((pos_vec.cpu(), neg_vec.cpu()), save_path)
    print(f"\nSaved vectors to: {save_path}")