import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN
from esm.utils.decoding import decode_sequence

from utils.gen_utils import sample_top_p

def get_esmc_layer_and_feature_dim():
    return (36, 1152)

def load_esmc_model(device='cuda'):
    tokenizer = get_esmc_model_tokenizers()
    model = ESMC.from_pretrained("B", device=torch.device(device)).to(torch.float32)
    return model, tokenizer

def extract_esmc_features(seqs, model, tokenizer, n_layer, batch_size=1, device='cuda'):
    layer_reps = [[] for _ in range(n_layer)]

    for start in range(0, len(seqs), batch_size):
        seq_batch = seqs[start:start + batch_size]
        
        seq_batch = [list(seq) for seq in seq_batch]
        x_batch = tokenizer.batch_encode_plus(
            seq_batch, 
            add_special_tokens=True, 
            padding=True, 
            is_split_into_words=True, 
            return_tensors="pt"
        )

        batch_lens = x_batch['attention_mask'].sum(dim=-1)

        with torch.no_grad():
            output = model(
                sequence_tokens=x_batch['input_ids'].to(device), 
                sequence_id=x_batch['attention_mask'].to(device)
            )
            representations = output.hidden_states

        for layer in range(n_layer):
            token_reps = representations[layer]
            for tokens, tokens_len in zip(token_reps, batch_lens):
                # Exclude special tokens at start and end
                layer_reps[layer].append(tokens[1:tokens_len-1].mean(0).cpu())

    # Stack representations: (num_layers, num_seqs, feature_dim)
    return torch.stack([torch.stack(reps) for reps in layer_reps])

def get_tokenwise_representations(tokens, masks, model):
    with torch.no_grad():
        output = model(
            sequence_tokens=tokens, 
            sequence_id=masks
        )
        representations = output.hidden_states
    
    return representations.permute(1, 2, 0, 3)[:, 1:-1] # batch size x sequence length x num layers x feature dim

def get_average_representation(tokens, masks, model):
    with torch.no_grad():
        output = model(
            sequence_tokens=tokens, 
            sequence_id=masks
        )
        representations = output.hidden_states
    return representations[-1][:, 1:-1].mean(dim=1)

def pred_tokens(tokens, model, steering_vectors=None, original_prediction=None, temperature=0.0, top_p=0.9):
    with torch.no_grad():
        if steering_vectors is not None:
            outputs = model.steering_forward(
                tokens.unsqueeze(0), 
                sequence_id=torch.ones_like(tokens).unsqueeze(0), 
                steering_vectors=steering_vectors)
        else:
            outputs = model(tokens.unsqueeze(0), sequence_id=torch.ones_like(tokens).unsqueeze(0))

    logits = outputs.sequence_logits

    if original_prediction is not None:
        mask  = F.one_hot(original_prediction, logits.size(-1))[:, 4:24].to(logits. device)

    logits = logits[0, :, 4:24]

    if original_prediction is not None:
        logits = logits + mask.float() * -1e8 # add negative logits to avoid predicting original tokens

    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)
        pred_seq = sample_top_p(probs, top_p)
    else:
        pred_seq = torch.argmax(logits, dim=-1)    
    pred_seq = pred_seq + 4

    pred_seq[0] = tokens[0]
    pred_seq[-1] = tokens[-1]

    return pred_seq

def generate_sequences(tokens, model, steering_vectors, masked_ratio, tokenizer, temperature=0.0, top_p=0.9):
    mask_idx = SEQUENCE_MASK_TOKEN
    tokens = tokens.clone()
    length = tokens.size(0) - 2
    candidate_sites = list(range(length))
    rounds = math.ceil(1.0 / masked_ratio)

    for _ in range(rounds):
        mask_size = min(math.ceil(length * masked_ratio), len(candidate_sites))
        if mask_size == 0:
            break

        indices = torch.randperm(len(candidate_sites))[:mask_size]
        mask_positions = torch.tensor([candidate_sites[i] for i in indices]) + 1  # +1 for offset
        candidate_sites = [site for i, site in enumerate(candidate_sites) if i not in indices]

        seq_token = tokens.clone()
        seq_token[mask_positions] = mask_idx
        new_seq = pred_tokens(seq_token, model, steering_vectors, temperature=temperature, top_p=top_p)
        tokens[mask_positions] = new_seq[mask_positions]

    return decode_sequence(tokens, tokenizer)