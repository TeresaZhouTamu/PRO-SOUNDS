import argparse
import pandas as pd
import numpy as np
from itertools import islice
from tqdm import tqdm
import types
import torch
import torch.nn.functional as F
import os
from collections import Counter

# Ensure these imports match your file structure
from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN
from utils.esmc_utils import decode_sequence, load_esmc_model, pred_tokens, get_tokenwise_representations

def get_max_homopolymer_length(seq_token):
    """
    Calculates the length of the longest consecutive run of the same token.
    """
    # Exclude BOS (0) and EOS (2) from the check
    tokens = seq_token[1:-1].tolist()
    if not tokens:
        return 0
    
    max_len = 1
    current_len = 1
    last_token = tokens[0]
    
    for t in tokens[1:]:
        if t == last_token:
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
            current_len = 1
            last_token = t
    
    # Check the final run
    if current_len > max_len:
        max_len = current_len
        
    return max_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== Inputs ===============================
    parser.add_argument('--data_path', type=str, required=True, 
                        help="Path to CSV file containing 'sequence' and 'label' columns.")
    parser.add_argument('--optimal_layer', type=int, required=True, 
                        help="The specific layer index used to identify mutation sites (e.g., 3).")

    # =========================== Settings ======================================
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--uncertainty_reweighted', action='store_true', help='Enable to use weighted steering vectors.')
    
    parser.add_argument('--uncertainty_type', type=str, choices=['total', 'AU', 'EU', 'Zscore'], default='total', 
                        help="Type of uncertainty used in vector generation.")

    parser.add_argument('--property', type=str, required=True, help="Property name (e.g., 'immunog').")
    parser.add_argument('--output_file', type=str, default=None, help="Path to save the optimized sequences.")

    parser.add_argument('--alpha', type=float, default=1.0, help="Steering strength coefficient (Applied during inference)")
    parser.add_argument('--sv_path', type=str, default=None, help="Direct path to a specific steering vector file (Overrides sv_from).")
    parser.add_argument('--sv_from', type=str, default="weighted_vectors", help="Folder path where steering vectors are saved (if sv_path is not set).")

    parser.add_argument('--n', type=int, default=1000, help="Maximal number of sequences to optimize.")
    parser.add_argument('--round', type=int, default=1, help="Number of optimization rounds.")
    parser.add_argument('--T', type=int, default=1, help="Number of mutation sites per round.")
    
    # === Refinement & Safeguard Settings ===
    parser.add_argument('--adapt_threshold', type=float, default=0.70, 
                        help="Confidence threshold. Sites with prob < this will be masked and re-predicted without steering.")
    parser.add_argument('--max_consecutive', type=int, default=3, 
                        help="Max allowed consecutive identical residues. Reverts refinement if exceeded.")
    
    args = parser.parse_args() 

    # 1. Load and Filter Data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
        
    print(f"Loading sequences from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    if 'sequence' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'sequence' and 'label' columns.")

    # Filter: Optimize Negative samples (0) -> Positive (1)
    neg_df = df[df['label'] == 0]
    org_seqs = neg_df['sequence'].to_list()
    
    print(f"Found {len(df)} total sequences.")
    print(f"Filtered to {len(org_seqs)} Negative sequences (Label=0) for optimization.")
    
    if len(org_seqs) == 0:
        print("Warning: No negative sequences found. Nothing to optimize.")
        exit(0)
    
    # 2. Load Model
    model, tokenizer = load_esmc_model(device=args.device)
    
    try:
        from module.steerable_esmc import steering_forward, esmc_steering_forward
        model.transformer.steering_forward = types.MethodType(steering_forward, model.transformer)
        model.steering_forward = types.MethodType(esmc_steering_forward, model)
    except ImportError:
        print("Error: Could not import steering methods from module.steerable_esmc.")
        exit(1)
    
    # 3. Load Steering Vectors
    if args.sv_path:
        steering_vectors_path = args.sv_path
    elif args.uncertainty_reweighted:
        steering_vectors_path = f"{args.sv_from}/ESMC_{args.property}_{args.uncertainty_type}_weighted_steering_vectors.pt"
    else:
        steering_vectors_path = f"{args.sv_from}/ESMC_{args.property}_steering_vectors.pt"
    
    print(f"Loading steering vectors from: {steering_vectors_path}")
    if not os.path.exists(steering_vectors_path):
        raise FileNotFoundError(f"Steering vector file not found at: {steering_vectors_path}")

    data = torch.load(steering_vectors_path, map_location=args.device)
    
    if isinstance(data, dict) and 'hybrid_vector' in data:
        print("  > Detected Hybrid Vector format.")
        steering_vectors = data['hybrid_vector']
    elif isinstance(data, (tuple, list)) and len(data) == 2:
        print("  > Detected Standard (Pos, Neg) Tuple format.")
        pos_steering_vectors, neg_steering_vectors = data
        steering_vectors = pos_steering_vectors - neg_steering_vectors
    else:
        raise ValueError(f"Unknown vector format in {steering_vectors_path}. Expected Tuple or Dict.")

    steering_vectors = steering_vectors.to(args.device)
    steering_vectors = steering_vectors * args.alpha
    
    # Create Zero Vectors for Unsteered Refinement
    zero_steering_vectors = torch.zeros_like(steering_vectors)
    
    layer_scoring_vec = steering_vectors[args.optimal_layer].clone()

    # List to store results from rounds
    new_seqs = [[] for _ in range(args.round)]

    print(f"Starting optimization using Optimal Layer: {args.optimal_layer}")
    print(f"Refinement Threshold: {args.adapt_threshold}, Max Homopolymer: {args.max_consecutive}")

    with torch.no_grad():
        for seq in tqdm(islice(org_seqs, args.n), total=min(len(org_seqs), args.n)):
            
            raw_ids = tokenizer.encode(seq, add_special_tokens=False)
            
            bos_id = getattr(tokenizer, 'bos_token_id', 0) or 0
            eos_id = getattr(tokenizer, 'eos_token_id', 2) or 2
            
            token_ids = [bos_id] + raw_ids + [eos_id]
            seq_token = torch.tensor(token_ids, dtype=torch.int64, device=args.device)
            
            prev_seq_token = seq_token.clone()
            prev_mut_sites = set()

            for r in range(args.round):
                # ==========================================================
                # Steered Mutation
                # ==========================================================
                
                # A. Get features
                full_features = get_tokenwise_representations(
                    prev_seq_token.unsqueeze(0), 
                    torch.ones_like(prev_seq_token).unsqueeze(0), 
                    model
                )
                
                layer_features = full_features[0, :, args.optimal_layer, :]

                # B. Scoring
                valid_features = layer_features[1:-1] # Skip BOS/EOS
                related_score = F.cosine_similarity(valid_features, layer_scoring_vec.unsqueeze(0), dim=-1).cpu().numpy()

                # C. Select Mutation Sites
                sorted_indices = np.argsort(related_score)
                
                mut_sites = []
                for idx in sorted_indices:
                    if len(mut_sites) >= args.T:
                        break
                    if idx not in prev_mut_sites:
                        mut_sites.append(idx)
                
                # Update tracking set
                prev_mut_sites.update(mut_sites)
                mut_sites_tensor = torch.LongTensor(mut_sites).to(args.device) + 1  

                # D. Mask and Steer
                masked_seq = prev_seq_token.clone()
                masked_seq[mut_sites_tensor] = SEQUENCE_MASK_TOKEN
                
                new_seq_token = pred_tokens(
                    masked_seq, 
                    model, 
                    steering_vectors, # Active Steering
                    original_prediction=prev_seq_token, 
                    temperature=0.0
                )
                
                # Construct the "Steered-Only" sequence
                # Only accept changes at the mutation sites
                steered_only_seq = masked_seq.clone()
                steered_only_seq[mut_sites_tensor] = new_seq_token[mut_sites_tensor]
                
                # Is the Steered sequence safe?
                steered_max_len = get_max_homopolymer_length(steered_only_seq)
                steered_is_safe = steered_max_len <= args.max_consecutive

                # ==========================================================
                # Unsteered Refinement 
                # ==========================================================
                
                # 1. Check Confidence of the Steered Sequence
                output = model(steered_only_seq.unsqueeze(0))
                
                if hasattr(output, 'sequence_logits'):
                    logits = output.sequence_logits
                else:
                    logits = output.logits
                logits = logits.squeeze(0) # [Seq, Vocab]
                
                # 2. Identify Low Confidence Positions
                probs = F.softmax(logits, dim=-1)
                current_indices = steered_only_seq.unsqueeze(1)
                current_probs = probs.gather(1, current_indices).squeeze(1)
                
                low_conf_mask = current_probs < args.adapt_threshold
                
                # 3. NOT touch mutated sites (current or previous rounds)
                if prev_mut_sites:
                    protected_sites_list = list(prev_mut_sites)
                    protected_indices = torch.LongTensor(protected_sites_list).to(args.device) + 1 # +1 for BOS
                    low_conf_mask[protected_indices] = False
                
                low_conf_mask[0] = False
                low_conf_mask[-1] = False
                
                # 4. Refill with UNSTEERED prediction
                candidate_seq = steered_only_seq.clone()
                
                if low_conf_mask.any():
                    refine_masked_seq = steered_only_seq.clone()
                    refine_masked_seq[low_conf_mask] = SEQUENCE_MASK_TOKEN
                    
                    # Predict using Standard ESMC
                    refined_tokens = pred_tokens(
                        refine_masked_seq,
                        model,
                        zero_steering_vectors, # NO Steering
                        original_prediction=steered_only_seq,
                        temperature=0.7
                    )
                    
                    candidate_seq[low_conf_mask] = refined_tokens[low_conf_mask]
                
                # ==========================================================
                # Safeguard Check
                # ==========================================================
                
                # Is the Refined sequence safe?
                cand_max_len = get_max_homopolymer_length(candidate_seq)
                cand_is_safe = cand_max_len <= args.max_consecutive
                
                # If Steered was Safe AND Refinement made it Unsafe -> REVERT
                # Otherwise (Steered was already Unsafe, OR Refinement is Safe) -> KEEP Refinement
                if steered_is_safe and not cand_is_safe:
                    # Refinement introduced a NEW violation. Reject it.
                    prev_seq_token = steered_only_seq
                elif not cand_is_safe:
                    prev_seq_token = steered_only_seq
                else:
                    # Accept the refinement
                    prev_seq_token = candidate_seq

                # Save result for this round
                new_seqs[r].append(decode_sequence(prev_seq_token, tokenizer))
                
    # 4. Save Results
    last_round_idx = args.round - 1
    final_seqs = new_seqs[last_round_idx]
    
    final_epochs = [args.round] * len(final_seqs)

    res_df = pd.DataFrame({'sequence': final_seqs, 'epoch': final_epochs})
    
    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        res_df.to_csv(args.output_file, index=False)
        print(f'Generated {len(res_df)} sequences (Round {args.round} only) saved to: {args.output_file}')
    else:
        print("Optimization complete (no output file specified).")