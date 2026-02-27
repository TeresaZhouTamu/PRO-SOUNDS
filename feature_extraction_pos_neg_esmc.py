import argparse
import pandas as pd
import torch
import os
import esm
from typing import List
from utils.esmc_utils import get_esmc_layer_and_feature_dim, load_esmc_model, extract_esmc_features

def get_esmc_model_name(model_size):
    """
    Returns the ESMC model name based on the specified size.
    """
    model_name_dict = {
        "300M": "esmc_300m",
        "600M": "esmc_600m"
    }
    
    if model_size in model_name_dict:
        return model_name_dict[model_size]
    else:
        raise ValueError(f"Unknown ESMC model size: {model_size}. Supported: 300M, 600M.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts ALL ESMC layer activations using user-provided utils.")
    
    parser.add_argument('--sample_data_path', type=str, required=True, 
                        help="Path to the CSV file containing 'sequence' and 'label' columns.")
    
    parser.add_argument('--property', type=str, required=True, help="Property name (e.g., 'immunog').")
    
    parser.add_argument('--num_data', type=int, default=None, help="Optional: Max sequences to process per class to save time.")
    parser.add_argument('--model', type=str, default="600M", help="ESMC size (Ignored if load_esmc_model is hardcoded).")
    parser.add_argument('--save_folder', type=str, default="probe_data_esmc", help="Output folder.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    
    try:
        print("Loading ESMC model using user-provided function...")
        model, tokenizer = load_esmc_model(device=args.device)
    except TypeError as e:
        print(f"Error calling load_esmc_model: {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    n_layers, feature_dim = get_esmc_layer_and_feature_dim()
    print(f"Model loaded. Config: {n_layers} layers, {feature_dim} dim.")
    print(f"Reading data from {args.sample_data_path}...")
    sample_df = pd.read_csv(args.sample_data_path)

    if 'sequence' not in sample_df.columns or 'label' not in sample_df.columns:
        raise ValueError("CSV must contain 'sequence' and 'label' columns.")

    pos_df = sample_df[sample_df['label'] == 1]
    neg_df = sample_df[sample_df['label'] == 0]

    if args.num_data:
        pos_df = pos_df.head(args.num_data)
        neg_df = neg_df.head(args.num_data)

    pos_seqs = pos_df['sequence'].tolist()
    neg_seqs = neg_df['sequence'].tolist()
    
    print(f"Found {len(pos_seqs)} positive (Label=1) and {len(neg_seqs)} negative (Label=0) sequences.")

    if len(pos_seqs) == 0 or len(neg_seqs) == 0:
        print("Warning: One of your classes is empty! Check your dataset labels.")

    print("\n--- Extracting Positive Features ---")
    pos_features_tensor = extract_esmc_features(pos_seqs, model, tokenizer, n_layers, args.batch_size, args.device)
    
    print("\n--- Extracting Negative Features ---")
    neg_features_tensor = extract_esmc_features(neg_seqs, model, tokenizer, n_layers, args.batch_size, args.device)
    
    pos_list_out = [pos_features_tensor[i].cpu() for i in range(n_layers)]
    neg_list_out = [neg_features_tensor[i].cpu() for i in range(n_layers)]

    pos_save_path = f"{args.save_folder}/esmc_600m_{args.property}_pos_features_list.pt"
    neg_save_path = f"{args.save_folder}/esmc_600m_{args.property}_neg_features_list.pt"
    
    torch.save(pos_list_out, pos_save_path)
    torch.save(neg_list_out, neg_save_path)
    
    print("\n--- ESMC Feature Extraction Complete ---")
    print(f"Positive features saved to: {pos_save_path}")
    print(f"Negative features saved to: {neg_save_path}")