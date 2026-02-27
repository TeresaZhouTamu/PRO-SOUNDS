import argparse
import pandas as pd
import torch
import os
import numpy as np
from utils.esmc_utils import get_esmc_layer_and_feature_dim, load_esmc_model, extract_esmc_features

def compute_robust_weights(uncertainties, clip_percentile=10):
    """
    Computes normalized inverse-variance weights.
    1. Clip the lowest X% of uncertainties to prevent massive weights (outliers).
    2. Compute Inverse Variance (1 / sigma^2).
    3. Normalize to sum to 1.
    """
    vals = np.array(uncertainties, dtype=np.float32)
    vals = np.nan_to_num(vals, nan=1.0) 
    
    # Winsorize (Clip) the lower tail
    limit = np.percentile(vals, clip_percentile)
    if limit == 0: limit = 1e-6
    vals_clipped = np.maximum(vals, limit)
    
    # 1 / Variance
    raw_weights = 1.0 / (vals_clipped ** 2)
    normalized_weights = raw_weights / np.sum(raw_weights)
    
    return normalized_weights

def compute_weighted_mean(features, weights):
    """
    Computes weighted mean of features across the sample dimension.
    Inputs:
        features: Tensor of shape (num_layers, num_samples, feature_dim)
        weights: Numpy array or Tensor of shape (num_samples,)
    Outputs:
        Tensor of shape (num_layers, feature_dim)
    """

    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, device=features.device, dtype=features.dtype)
    else:
        weights = weights.to(features.device).to(features.dtype)
        
    w_reshaped = weights.view(1, -1, 1)
    w_sum = w_reshaped.sum(dim=1, keepdim=True)
    w_normalized = w_reshaped / (w_sum + 1e-8)
    
    # Weighted Sum: sum(w_i * x_i)
    weighted_mean = (features * w_normalized).sum(dim=1)
    
    return weighted_mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to CSV with 'sequence' and 'label' columns.")
    parser.add_argument('--uncertainty_path', type=str, required=True, help="Path to CSV with 'sequence' and uncertainty columns.")
    parser.add_argument('--uncertainty_type', type=str, choices=['total', 'aleatoric', 'epistemic'], default='EU', 
                        help="Type of uncertainty to use for weighting: 'total', 'AU' (Aleatoric), or 'EU' (Epistemic).")
    parser.add_argument('--property', type=str, required=True, help="Property name (e.g., 'sol').")
    parser.add_argument('--num_data', type=int, default=None, help="Max sequences to process per group.")
    parser.add_argument('--save_folder', type=str, default="weighted_vectors", help="Folder to save vectors.")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use.")
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    col_map = {
        'total': 'total_uncertainty',
        'AU': 'aleatoric_uncertainty',
        'EU': 'epistemic_uncertainty'
    }
    
    selected_col = col_map[args.uncertainty_type]

    # Load and Merge Data
    print(f"Loading data from {args.data_path}...")
    df_data = pd.read_csv(args.data_path)
    
    print(f"Loading uncertainty scores from {args.uncertainty_path}...")
    df_unc = pd.read_csv(args.uncertainty_path)

    if 'sequence' not in df_data.columns or 'label' not in df_data.columns:
        raise ValueError("Data CSV must contain 'sequence' and 'label' columns.")
    
    if 'sequence' not in df_unc.columns:
        raise ValueError("Uncertainty CSV must contain a 'sequence' column for alignment.")
    
    if selected_col not in df_unc.columns:
        print(f"Error: The column '{selected_col}' was not found in {args.uncertainty_path}.")
        print(f"Available columns: {list(df_unc.columns)}")
        exit(1)
        
    print(f"Merging using uncertainty column: {selected_col}")
    merged_df = pd.merge(df_data, df_unc[['sequence', selected_col]], on='sequence', how='inner')
    
    print(f"Merged dataset size: {len(merged_df)} sequences.")

    pos_df = merged_df[merged_df['label'] == 1].copy()
    neg_df = merged_df[merged_df['label'] == 0].copy()
    
    if args.num_data is not None:
        pos_df = pos_df.head(args.num_data)
        neg_df = neg_df.head(args.num_data)
        
    pos_seqs = pos_df['sequence'].tolist()
    neg_seqs = neg_df['sequence'].tolist()
    
    print(f"Processing {len(pos_seqs)} Positive and {len(neg_seqs)} Negative sequences.")

    if len(pos_seqs) == 0 or len(neg_seqs) == 0:
        raise ValueError("One of the groups is empty! Check your 'label' column (should be 0 or 1) and your merge logic.")

    print("\n--- Calculating Robust Weights ---")
    pos_weights = compute_robust_weights(pos_df[selected_col].values)
    neg_weights = compute_robust_weights(neg_df[selected_col].values)

    # # DIAGNOSTICS: Check Effective Sample Size (ESS)
    # # ESS = 1 / sum(w^2). Indicates how many samples are effectively contributing.
    # pos_ess = 1.0 / np.sum(pos_weights ** 2)
    # neg_ess = 1.0 / np.sum(neg_weights ** 2)
    
    # print(f"Positive Group ESS: {pos_ess:.1f} (out of {len(pos_seqs)})")
    # print(f"Negative Group ESS: {neg_ess:.1f} (out of {len(neg_seqs)})")
    
    # if pos_ess < 5.0 or neg_ess < 5.0:
    #     print("WARNING: ESS is extremely low. A few sequences are dominating the steering vector.")
    # print("----------------------------------\n")

    # 4. Load Model
    n_layers, feature_dim = get_esmc_layer_and_feature_dim()
    try:
        model, tokenizer = load_esmc_model(device=args.device)
    except Exception as e:
        print(f"Error loading ESMC model: {e}")
        exit(1)

    print("Extracting Positive Features...")
    pos_seq_repr_mat = extract_esmc_features(pos_seqs, model, tokenizer, n_layers, batch_size=args.batch_size, device=args.device)
    print("Extracting Negative Features...")
    neg_seq_repr_mat = extract_esmc_features(neg_seqs, model, tokenizer, n_layers, batch_size=args.batch_size, device=args.device)

    print(f"\nComputing {args.uncertainty_type}-Weighted Steering Vectors...")
    pos_steering_vectors = compute_weighted_mean(pos_seq_repr_mat, pos_weights)
    neg_steering_vectors = compute_weighted_mean(neg_seq_repr_mat, neg_weights)
    pos_steering_vectors = pos_steering_vectors.detach().cpu()
    neg_steering_vectors = neg_steering_vectors.detach().cpu()

    os.makedirs(args.save_folder, exist_ok=True)    
    save_path = f"{args.save_folder}/ESMC_{args.property}_{args.uncertainty_type}_weighted_steering_vectors.pt"
    torch.save((pos_steering_vectors, neg_steering_vectors), save_path)
    print(f"\nWeighted vectors saved to: {save_path}")