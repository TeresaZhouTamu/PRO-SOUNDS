import argparse
import warnings
import torch
import os
import sys
import pandas as pd # Added for CSV
import numpy as np
import torch.nn.functional as F # Added for activations
sys.path.append(os.getcwd())
import wandb
import random
import json
import re
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import logging
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef
from torchmetrics.regression import SpearmanCorrCoef
from accelerate import Accelerator
from accelerate.utils import set_seed
from time import strftime, localtime
from datasets import load_dataset
from transformers import EsmModel, BertModel
from transformers import EsmTokenizer, EsmForMaskedLM, BertForMaskedLM, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer
from src.utils.data_utils import BatchSampler
from src.models.adapter import AdapterModel
from src.utils.metrics import MultilabelF1Max
from src.data.get_esm3_structure_seq import VQVAE_SPECIAL_TOKENS

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ==========================================
# 1. Evidential Deep Learning Loss Functions
# ==========================================

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def kl_divergence(alpha, num_classes, device=None):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def loglikelihood_loss(y, alpha, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    log_likelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    log_likelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    return log_likelihood_err + log_likelihood_var

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    
    # Label smoothing-like One-Hot adjustment
    # A = torch.sum((y - m) ** 2, dim=1, keepdim=True)
    # Standard MSE for EDL
    A = torch.sum((y - m)**2, dim=1, keepdim=True) + torch.sum(alpha*(S-alpha)/(S*S*(S+1)), dim=1, keepdim=True)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl = kl_divergence(kl_alpha, num_classes, device=device)
    
    annealing_coef = min(1, epoch_num / annealing_step)
    return (A + annealing_coef * kl).mean()

class EDLLoss(nn.Module):
    def __init__(self, num_classes, annealing_step=10, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.device = device

    def forward(self, output, target, epoch_num):
        # Target must be one-hot encoded for EDL MSE
        if target.dim() == 1:
            target = F.one_hot(target, num_classes=self.num_classes).float()
        
        # Convert logits to evidence
        evidence = softplus_evidence(output)
        alpha = evidence + 1
        
        loss = mse_loss(target, alpha, epoch_num, self.num_classes, self.annealing_step, self.device)
        return loss

# ==========================================
# 2. Main Logic
# ==========================================

def min_max_normalize_dataset(train_dataset, val_dataset, test_dataset):
    # This might not be necessary for simple 0/1 classification labels, 
    # but kept for compatibility with existing code structure
    labels = [e["label"] for e in train_dataset]
    min_label, max_label = min(labels), max(labels)
    # Check if normalization is actually needed (e.g., if labels are not already 0-1)
    if max_label > 1:
        for e in train_dataset: e["label"] = (e["label"] - min_label) / (max_label - min_label)
        for e in val_dataset: e["label"] = (e["label"] - min_label) / (max_label - min_label)
        for e in test_dataset: e["label"] = (e["label"] - min_label) / (max_label - min_label)
    return train_dataset, val_dataset, test_dataset


def train(args, model, plm_model, accelerator, metrics_dict, train_loader, val_loader, test_loader, 
          loss_fn, optimizer, device):
    best_val_loss, best_val_metric_score = float("inf"), -float("inf")
    val_loss_list, val_metric_list = [], []
    path = os.path.join(args.ckpt_dir, args.model_name)
    global_steps = 0
    
    for epoch in range(args.max_train_epochs):
        print(f"---------- Epoch {epoch} ----------")
        # train
        model.train()
        epoch_train_loss = 0
        epoch_iterator = tqdm(train_loader)
        
        for batch in epoch_iterator:
            with accelerator.accumulate(model):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                
                label = batch["label"]
                logits = model(plm_model, batch)
                
                # EDL Training Step
                if isinstance(loss_fn, EDLLoss):
                    loss = loss_fn(logits, label, epoch)
                else:
                    loss = loss_fn(logits, label)
                
                epoch_train_loss += loss.item() * len(label)                
                global_steps += 1
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_postfix(train_loss=loss.item())
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "train/epoch": epoch}, step=global_steps)
                    
        train_loss = epoch_train_loss / len(train_loader.dataset)
        print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f}')
        
        # eval every epoch
        model.eval()
        with torch.no_grad():
            val_loss, val_metric_dict = eval_loop(args, model, plm_model, metrics_dict, val_loader, loss_fn, device, epoch)
            val_metric_score = val_metric_dict[args.monitor]
            val_metric_list.append(val_metric_score)
            val_loss_list.append(val_loss)
            
            if args.wandb:
                val_log = {"valid/loss": val_loss}
                for metric_name, metric_score in val_metric_dict.items():
                    val_log[f"valid/{metric_name}"] = metric_score
                wandb.log(val_log)
            print(f'EPOCH {epoch} VAL loss: {val_loss:.4f} {args.monitor}: {val_metric_score:.4f}')
    
        # early stopping logic (Unchanged)
        if args.monitor == "loss":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path)
                print(f'>>> BEST at epoch {epoch}, loss: {best_val_loss:.4f}')
            
            if len(val_loss_list) - val_loss_list.index(min(val_loss_list)) > args.patience:
                print(f'>>> Early stopping at epoch {epoch}')
                break
        else:
            if val_metric_score > best_val_metric_score:
                best_val_metric_score = val_metric_score
                torch.save(model.state_dict(), path)
                print(f'>>> BEST at epoch {epoch}, {args.monitor}: {best_val_metric_score:.4f}')
            
            if len(val_metric_list) - val_metric_list.index(max(val_metric_list)) > args.patience:
                print(f'>>> Early stopping at epoch {epoch}')
                break
    
    print(f"TESTING: loading from {path}")
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        # Pass epoch=max_epoch for testing loss calc
        test_loss, test_metric_dict = eval_loop(args, model, plm_model, metrics_dict, test_loader, loss_fn, device, args.max_train_epochs)
        test_metric_score = test_metric_dict[args.monitor]
        
        if args.wandb:
            test_log = {"test/loss": test_loss}
            for metric_name, metric_score in test_metric_dict.items():
                test_log[f"test/{metric_name}"] = metric_score
            wandb.log(test_log)
        print(f'EPOCH {epoch} TEST loss: {test_loss:.4f} {args.monitor}: {test_metric_score:.4f}')
        for metric_name, metric_score in test_metric_dict.items():
            print(f'>>> {metric_name}: {metric_score:.4f}')

def eval_loop(args, model, plm_model, metrics_dict, dataloader, loss_fn, device=None, epoch=0):
    total_loss = 0
    epoch_iterator = tqdm(dataloader)
    
    # Store uncertainties for stats
    all_eu = []
    all_au = []
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        logits = model(plm_model, batch)
        
        # --- EDL Processing ---
        # 1. Calculate Evidence
        evidence = softplus_evidence(logits)
        # 2. Calculate Alpha
        alpha = evidence + 1
        # 3. Strength (S)
        S = torch.sum(alpha, dim=1, keepdim=True)
        # 4. Expected Probability (p = alpha / S)
        prob = alpha / S
        
        # --- Uncertainty Calculation ---
        # Epistemic Uncertainty (EU) = K / S (Vacuum Uncertainty)
        # Note: Some papers use num_classes / S, others use K / S.
        eu = args.num_labels / S
        
        # Aleatoric Uncertainty (AU) = Entropy of Expected Distribution
        # AU = - sum(p * log(p))
        au = -torch.sum(prob * torch.log(prob + 1e-8), dim=1, keepdim=True)
        
        all_eu.append(eu.detach().cpu())
        all_au.append(au.detach().cpu())

        # --- Loss ---
        if isinstance(loss_fn, EDLLoss):
            loss = loss_fn(logits, label, epoch)
        else:
            loss = loss_fn(logits, label)
            
        # --- Metrics ---
        # Torchmetrics expect probabilities or logits. 
        # Since we are doing EDL, we pass the EXPECTED PROBABILITY (prob).
        for metric_name, metric in metrics_dict.items():
            # Adjust metric input based on type
            if args.problem_type == 'multi_label_classification':
                 # Not strictly supported by standard Dirichlet EDL, usually uses Beta density
                metric(logits, label.float())
            else:
                if args.num_labels == 2:
                    # For binary metrics, usually expect probability of positive class
                    metric(prob[:, 1], label)
                else:
                    metric(prob, label)
                    
        total_loss += loss.item() * len(label)
        epoch_iterator.set_postfix(eval_loss=loss.item())
    
    # Calculate Mean Uncertainties
    avg_eu = torch.cat(all_eu).mean().item()
    avg_au = torch.cat(all_au).mean().item()

    metrics_result_dict = {}
    epoch_loss = total_loss / len(dataloader.dataset)
    for metric_name, metric in metrics_dict.items():
        metrics_result_dict[metric_name] = metric.compute().item()
        metric.reset()
    
    # Add Uncertainty to result dict
    metrics_result_dict['avg_epistemic_uncertainty'] = avg_eu
    metrics_result_dict['avg_aleatoric_uncertainty'] = avg_au
    metrics_result_dict['loss'] = epoch_loss
    
    return epoch_loss, metrics_result_dict

# ==========================================
# 3. Data Processing (CSV Support)
# ==========================================

def process_data_line(data, args):
    if args.max_seq_len is not None:
        data["aa_seq"] = data["aa_seq"][:args.max_seq_len]
        # Handle structure seqs if present
        if args.structure_seqs:
            for seq in args.structure_seqs:
                if seq in data:
                    data[seq] = data[seq][:args.max_seq_len]
        token_num = min(len(data["aa_seq"]), args.max_seq_len)
    else:
        token_num = len(data["aa_seq"])
    return data, token_num

def process_dataset_from_csv(file_path, args):
    """
    Parses CSV with columns: protein_name,label,cluster_id,aa_seq
    """
    df = pd.read_csv(file_path)
    dataset, token_nums = [], []
    
    for _, row in df.iterrows():
        # Create dict structure expected by collate_fn
        data = {
            "aa_seq": row["aa_seq"],
            "label": int(row["label"]),
            # Optional: Keep metadata
            # "protein_name": row["protein_name"],
            # "cluster_id": row["cluster_id"]
        }
        
        # Handle Multilabel (if comma separated string in CSV)
        if args.problem_type == 'multi_label_classification':
            raise NotImplementedError("EDL code below is optimized for Multi-Class (Dirichlet), not Multi-Label.")
            
        data, token_num = process_data_line(data, args)
        dataset.append(data)
        token_nums.append(token_num)
        
    return dataset, token_nums

# ==========================================
# 4. Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model params
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0)
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--pooling_method', type=str, default='attention1d')
    parser.add_argument('--return_attentions', action='store_true')
    parser.add_argument('--pooling_dropout', type=float, default=0.25)
    
    # dataset
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--num_labels', type=int, default=2) # Default binary
    parser.add_argument('--problem_type', type=str, default="single_label_classification")
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--metrics', type=str, default="accuracy,auc,f1")
    
    # train model
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_batch_token', type=int, default=10000)
    parser.add_argument('--max_train_epochs', type=int, default=20)
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--monitor', type=str, default="loss") # Monitor loss for EDL usually safer
    parser.add_argument('--structure_seqs', type=str, default=None)
    
    # EDL specific
    parser.add_argument('--use_edl', action='store_true', default=True, help="Use Evidential Deep Learning")
    parser.add_argument('--edl_annealing_step', type=int, default=10, help="Epochs to anneal KL divergence")

    # save/log
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--ckpt_root', default="ckpt")
    parser.add_argument('--ckpt_dir', default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='SES-Adapter-EDL')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.structure_seqs is not None:
        args.structure_seqs = args.structure_seqs.split(',')
    else:
        args.structure_seqs = []

    # Load Config if exists (Optional overwrite)
    if args.dataset_config:
        dataset_config = json.loads(open(args.dataset_config).read())
        # ... (Config loading logic from original code) ...
        # Simplified here for brevity, ensure args are set correctly

    # Metrics Setup
    metrics_dict = {}
    if args.metrics != 'None':
        metric_names = args.metrics.split(',')
        for m in metric_names:
            if m == 'accuracy':
                metrics_dict[m] = BinaryAccuracy() if args.num_labels == 2 else Accuracy(task="multiclass", num_classes=args.num_labels)
            elif m == 'auc':
                metrics_dict[m] = BinaryAUROC() if args.num_labels == 2 else AUROC(task="multiclass", num_classes=args.num_labels)
            elif m == 'f1':
                metrics_dict[m] = BinaryF1Score() if args.num_labels == 2 else F1Score(task="multiclass", num_classes=args.num_labels)
        for metric_name, metric in metrics_dict.items():
            metric.to(device)

    # Init WandB
    if args.ckpt_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.ckpt_dir = os.path.join(args.ckpt_root, current_date)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    if args.wandb:
        if args.wandb_run_name is None: args.wandb_run_name = f"Adapter-EDL-{args.dataset}"
        if args.model_name is None: args.model_name = f"{args.wandb_run_name}.pt"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity, config=vars(args))

    # Load Model

    if "esm" in args.plm_model:
        print(f"Loading ESM model: {args.plm_model}")
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        plm_model = EsmModel.from_pretrained(args.plm_model, output_hidden_states=True).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "bert" in args.plm_model:
        print(f"Loading BERT model: {args.plm_model}")
        tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = BertModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "prot_t5" in args.plm_model:
        print(f"Loading ProtT5 model: {args.plm_model}")
        tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    elif "ankh" in args.plm_model:
        print(f"Loading Ankh model: {args.plm_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    # ... (other PLM loaders) ...
    
    if args.structure_seqs is not None and 'esm3_structure_seq' in args.structure_seqs: 
         args.vocab_size = max(plm_model.config.vocab_size, 4100)
    else:
         args.vocab_size = plm_model.config.vocab_size

    model = AdapterModel(args)
    model.to(device)

    # --- Load Data using New CSV Function ---
    print(f"Loading CSV data from: {args.train_file}")
    train_dataset, train_token_num = process_dataset_from_csv(args.train_file, args)
    
    if args.valid_file:
        val_dataset, val_token_num = process_dataset_from_csv(args.valid_file, args)
    else:
        # Split train if no val provided (placeholder logic)
        val_dataset, val_token_num = [], [] 

    if args.test_file:
        test_dataset, test_token_num = process_dataset_from_csv(args.test_file, args)
    else:
        test_dataset, test_token_num = [], []
    
    print(">>> trainset: ", len(train_dataset))
    print(">>> valset: ", len(val_dataset))
    print(">>> testset: ", len(test_dataset))
    print("---------- Smple 3 data point from trainset ----------")
    
    for i in random.sample(range(len(train_dataset)), 3):
        print(">>> ", train_dataset[i])

    def e_descriptor_embedding(aa_input_ids):
        aa_seqs = [tokenizer.convert_ids_to_tokens(aa_input_ids[i]) for i in range(len(aa_input_ids))]
        e1 = {'A': 0.008, 'R': 0.171, 'N': 0.255, 'D': 0.303, 'C': -0.132, 'Q': 0.149, 'E': 0.221, 'G': 0.218,
            'H': 0.023, 'I': -0.353, 'L': -0.267, 'K': 0.243, 'M': -0.239, 'F': -0.329, 'P': 0.173, 'S': 0.199,
            'T': 0.068, 'W': -0.296, 'Y': -0.141, 'V': -0.274}
        e2 = {'A': 0.134, 'R': -0.361, 'N': 0.038, 'D': -0.057, 'C': 0.174, 'Q': -0.184, 'E': -0.28, 'G': 0.562,
            'H': -0.177, 'I': 0.071, 'L': 0.018, 'K': -0.339, 'M': -0.141, 'F': -0.023, 'P': 0.286, 'S': 0.238,
            'T': 0.147, 'W': -0.186, 'Y': -0.057, 'V': 0.136}
        e3 = {'A': -0.475, 'R': 0.107, 'N': 0.117, 'D': -0.014, 'C': 0.07, 'Q': -0.03, 'E': -0.315, 'G': -0.024,
            'H': 0.041, 'I': -0.088, 'L': -0.265, 'K': -0.044, 'M': -0.155, 'F': 0.072, 'P': 0.407, 'S': -0.015,
            'T': -0.015, 'W': 0.389, 'Y': 0.425, 'V': -0.187}
        e4 = {'A': -0.039, 'R': -0.258, 'N': 0.118, 'D': 0.225, 'C': 0.565, 'Q': 0.035, 'E': 0.157, 'G': 0.018,
            'H': 0.28, 'I': -0.195, 'L': -0.274, 'K': -0.325, 'M': 0.321, 'F': -0.002, 'P': -0.215, 'S': -0.068,
            'T': -0.132, 'W': 0.083, 'Y': -0.096, 'V': -0.196}
        e5 = {'A': 0.181, 'R': -0.364, 'N': -0.055, 'D': 0.156, 'C': -0.374, 'Q': -0.112, 'E': 0.303, 'G': 0.106,
            'H': -0.021, 'I': -0.107, 'L': 0.206, 'K': -0.027, 'M': 0.077, 'F': 0.208, 'P': 0.384, 'S': -0.196,
            'T': -0.274, 'W': 0.297, 'Y': -0.091, 'V': -0.299}
        # Build descriptor tensors
        descriptor_dicts = [e1, e2, e3, e4, e5]
        descriptors = {}
        for aa in e1.keys():
            descriptors[aa] = [d[aa] for d in descriptor_dicts]   # Each amino acid corresponds to a 5-dimensional descriptor
        e_embeds = []
        for seq in aa_seqs:
            seq_embeds = [descriptors.get(aa, [0.0]*5) for aa in seq]
            e_embeds.append(seq_embeds)
        e_embeds = torch.tensor(e_embeds).float()
        return e_embeds

    def z_descriptor_embedding(aa_input_ids):
        aa_seqs = [tokenizer.convert_ids_to_tokens(aa_input_ids[i]) for i in range(len(aa_input_ids))]
        z1 = {'A': 0.07, 'R': 2.88, 'N': 3.22, 'D': 3.64, 'C': 0.71, 'Q': 2.18, 'E': 3.08, 'G': 2.23, 'H': 2.41,
            'I': -4.44, 'L': -4.19, 'K': 2.84, 'M': -2.49, 'F': -4.92, 'P': -1.22, 'S': 1.96, 'T': 0.92, 'W': -4.75,
            'Y': -1.39, 'V': -2.69}
        z2 = {'A': -1.73, 'R': 2.52, 'N': 1.45, 'D': 1.13, 'C': -0.97, 'Q': 0.53, 'E': 0.39, 'G': -5.36, 'H': 1.74,
            'I': -1.68, 'L': -1.03, 'K': 1.41, 'M': -0.27, 'F': 1.30, 'P': 0.88, 'S': -1.63, 'T': -2.09, 'W': 3.65,
            'Y': 2.32, 'V': -2.53}
        z3 = {'A': 0.09, 'R': -3.44, 'N': 0.84, 'D': 2.36, 'C': 4.13, 'Q': -1.14, 'E': -0.07, 'G': 0.30, 'H': 1.11,
            'I': -1.03, 'L': -0.98, 'K': -3.14, 'M': -0.41, 'F': 0.45, 'P': 2.23, 'S': 0.57, 'T': -1.40, 'W': 0.85,
            'Y': 0.01, 'V': -1.29}
        # Build descriptor tensors
        descriptor_dicts = [z1, z2, z3]
        descriptors = {}
        for aa in z1.keys():
            descriptors[aa] = [d[aa] for d in descriptor_dicts]
        z_embeds = []
        for seq in aa_seqs:
            seq_embeds = [descriptors.get(aa, [0.0]*3) for aa in seq]
            z_embeds.append(seq_embeds)
        z_embeds = torch.tensor(z_embeds).float()
        return z_embeds

    def aac_embedding(aa_input_ids):
        e_embeds = e_descriptor_embedding(aa_input_ids)  # Shape: (batch_size, seq_len, 5)
        z_embeds = z_descriptor_embedding(aa_input_ids)  # Shape: (batch_size, seq_len, 3)
        ez_embeds = torch.cat([e_embeds, z_embeds], dim=-1)  # Shape: (batch_size, seq_len, 8)
        batch_size, seq_len, k = ez_embeds.shape  # k = 8

        # Initialize a tensor to hold the autocovariance matrices
        covariances = []
        for l in range(seq_len):
            seq_len_l = seq_len - l
            x1 = ez_embeds[:, :seq_len_l, :]  # Shape: (batch_size, seq_len_l, k)
            x2 = ez_embeds[:, l:, :]          # Shape: (batch_size, seq_len_l, k)
            # Compute covariance matrices without explicit loops over features
            cov_l = torch.matmul(x1.transpose(1, 2), x2) / seq_len_l  # Shape: (batch_size, k, k)
            covariances.append(cov_l)

        # Stack and reshape the covariances
        covariances = torch.stack(covariances, dim=1)  # Shape: (batch_size, seq_len, k, k)
        covariances = covariances.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, k * k)
        return covariances.float()

    # Collate function (Need to ensure it matches the original logic, copied here for context)
    def collate_fn(examples):
        # # ... (Use the collate_fn provided in your original script) ...
        # # ... (Assuming it's available in scope or copied over) ...
        # # This part requires the original collate_fn logic, 
        # # but ensure 'label' is converted to LongTensor for classification
        # aa_seqs, labels = [], []
        # # ... (rest of collate logic) ...
        # # Simplified placeholder for the list comprehension:
        # for e in examples:
        #     aa_seqs.append(e["aa_seq"])
        #     labels.append(e["label"])
        
        # aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
        # # Fix labels for classification
        # labels = torch.as_tensor(labels, dtype=torch.long)
        # return {"aa_input_ids": aa_inputs["input_ids"], "attention_mask": aa_inputs["attention_mask"], "label": labels}

        aa_seqs, labels = [], []

        for e in examples:
            # idx_list.append(e['idx'])
            aa_seq = e["aa_seq"]

            # --- Tokenizer Specific Formatting ---
            if 'prot_bert' in args.plm_model or "prot_t5" in args.plm_model:
                # ProtBert/T5 usually require spaces between residues
                aa_seq = " ".join(list(aa_seq))
                aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
            elif 'ankh' in args.plm_model:
                # Ankh usually expects a list of characters
                aa_seq = list(aa_seq)
        
            aa_seqs.append(aa_seq)
            labels.append(e["label"])

        # --- Tokenization ---
        if 'ankh' in args.plm_model:
            aa_inputs = tokenizer.batch_encode_plus(
                aa_seqs, 
                add_special_tokens=True, 
                padding=True, 
                is_split_into_words=True, 
                return_tensors="pt"
            )
        else:
            aa_inputs = tokenizer(
                aa_seqs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
    
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
    
        # --- Label Processing ---
        if args.problem_type == 'regression':
            labels = torch.as_tensor(labels, dtype=torch.float)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)

        # --- Construct Final Dict ---
        data_dict = {
            "aa_input_ids": aa_input_ids, 
            "attention_mask": attention_mask, 
            "label": labels,
            # "idx": torch.tensor(idx_list, dtype=torch.long)
        }

        return data_dict

    # Setup Accelerator & Optimizer
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- EDL Loss Selection ---
    if args.use_edl:
        print(f"Using Evidential Deep Learning Loss (Annealing: {args.edl_annealing_step})")
        loss_fn = EDLLoss(num_classes=args.num_labels, annealing_step=args.edl_annealing_step, device=device)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Dataloaders
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, collate_fn=collate_fn, batch_sampler=BatchSampler(train_token_num, args.max_batch_token))
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, collate_fn=collate_fn, batch_sampler=BatchSampler(val_token_num, args.max_batch_token, False))
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, collate_fn=collate_fn, batch_sampler=BatchSampler(test_token_num, args.max_batch_token, False))

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)

    train(args, model, plm_model, accelerator, metrics_dict, train_loader, val_loader, test_loader, loss_fn, optimizer, device)
    
    if args.wandb: wandb.finish()