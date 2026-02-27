import argparse
import pandas as pd
import torch
import os
from utils.esmc_utils import get_esmc_layer_and_feature_dim, load_esmc_model, extract_esmc_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== Inputs ===============================
    parser.add_argument('--data_path', type=str, required=True, 
                        help="Path to CSV file containing 'sequence' and 'label' columns.")
    parser.add_argument('--property', type=str, required=True, 
                        help="Property name (e.g., 'immunog') used for saving the output file.")

    # =========================== Settings =============================
    parser.add_argument('--num_data', type=int, default=None, 
                        help="Number of sequences to use per class. If None, uses all available data.")
    parser.add_argument('--save_folder', type=str, default="saved_steering_vectors", 
                        help="Folder path to save steering vectors.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run calculations on (cuda/cpu).")

    args = parser.parse_args()

    # 1. Load Data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    if 'sequence' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'sequence' and 'label' columns.")

    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    pos_seqs = pos_df['sequence'].to_list()
    neg_seqs = neg_df['sequence'].to_list()
    
    print(f"Found {len(pos_seqs)} Positive (Label 1) and {len(neg_seqs)} Negative (Label 0) sequences.")
    if args.num_data is not None:
        pos_seqs = pos_seqs[:args.num_data]
        neg_seqs = neg_seqs[:args.num_data]
        print(f"Subsampled to {len(pos_seqs)} Positive and {len(neg_seqs)} Negative sequences.")

    if len(pos_seqs) == 0 or len(neg_seqs) == 0:
        raise ValueError("Error: One of the classes is empty. Cannot compute difference vector.")

    n_layers, feature_dim = get_esmc_layer_and_feature_dim()
    
    try:
        model, tokenizer = load_esmc_model(device=args.device)
    except TypeError:
        # Fallback if your specific load_esmc_model definition doesn't accept device arg
        model, tokenizer = load_esmc_model()
        model = model.to(args.device)

    print("Extracting features for Positive sequences...")
    pos_seq_repr_mat = extract_esmc_features(pos_seqs, model, tokenizer, n_layers)
    print("Extracting features for Negative sequences...")
    neg_seq_repr_mat = extract_esmc_features(neg_seqs, model, tokenizer, n_layers)

    print("Computing mean steering vectors...")
    pos_steering_vectors, neg_steering_vectors = [], []
    
    for i in range(n_layers):
        pos_vec = pos_seq_repr_mat[i].mean(dim=0)
        neg_vec = neg_seq_repr_mat[i].mean(dim=0)
        
        pos_steering_vectors.append(pos_vec)
        neg_steering_vectors.append(neg_vec)

    pos_steering_vectors = torch.stack(pos_steering_vectors).detach().cpu()
    neg_steering_vectors = torch.stack(neg_steering_vectors).detach().cpu()

    os.makedirs(args.save_folder, exist_ok=True)
    output_path = f"{args.save_folder}/ESMC_{args.property}_steering_vectors.pt"
    torch.save((pos_steering_vectors, neg_steering_vectors), output_path)
    print(f"Steering vectors saved to: {output_path}")