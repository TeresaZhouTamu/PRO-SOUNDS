import argparse
import warnings
import torch
import os
import sys
import pickle
sys.path.append(os.getcwd())
import wandb
import random
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
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
from src.utils.loss_fn import MultiClassFocalLossWithAlpha
from src.data.get_esm3_structure_seq import VQVAE_SPECIAL_TOKENS
from collections import defaultdict
import torch.nn.functional as F

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def min_max_normalize_dataset(train_dataset, val_dataset, test_dataset):
    labels = [e["label"] for e in train_dataset]
    min_label, max_label = min(labels), max(labels)
    for e in train_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in val_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in test_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    return train_dataset, val_dataset, test_dataset

def train(args, model, plm_model, accelerator, metrics_dict, train_loader, val_loader, test_loader, 
          loss_fn, optimizer, device):
    
    # --- FIX 1: Correctly initialize dynamics using actual item content ---
    train_dynamics = {}
    # We iterate over the dataset directly to access the specific dictionaries
    for item in train_loader.dataset:
        train_dynamics[item['idx']] = {
            "conf": [], 
            "corr": [], 
            "eu": [], 
            "au": [], 
            # "name": item['name']
        }

    num_examples = len(train_loader.dataset)
    print(f"Training on {num_examples} examples with {len(train_loader)} batches per epoch.")

    best_val_loss, best_val_metric_score = float("inf"), -float("inf")
    val_loss_list, val_metric_list = [], []
    path = os.path.join(args.ckpt_dir, args.model_name)
    global_steps = 0
    
    for epoch in range(args.max_train_epochs):
        print(f"---------- Epoch {epoch} ----------")
        model.train()
        epoch_train_loss = 0
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in epoch_iterator:
            with accelerator.accumulate(model):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                label = batch["label"]

                # 1. Forward Pass
                logits = model(plm_model, batch)

                # 2. Calculate Standard Loss
                if args.problem_type == 'multi_label_classification':
                    loss = loss_fn(logits, label.float())
                elif args.problem_type == 'regression':
                    loss = loss_fn(logits.squeeze(), label.squeeze())
                else:
                    # Single Label / Cross Entropy
                    loss = loss_fn(logits, label)

                # 3. Initialize Uncertainty Placeholders
                epistemic_uncertainty = None
                aleatoric_uncertainty = None
                gold_probs = None

                # 4. Calculate Uncertainty Metrics (Post-hoc Evidential)
                if args.problem_type not in ['regression', 'multi_label_classification']:
                    probs_softmax = torch.softmax(logits, dim=1)
                    gold_probs = probs_softmax[torch.arange(len(label)), label]
                    preds = torch.argmax(probs_softmax, dim=1)

                    evidence = F.softplus(logits)
                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    
                    epistemic_uncertainty = args.num_labels / S
                    prob_dist_edl = alpha / S
                    entropy = -torch.sum(prob_dist_edl * torch.log(prob_dist_edl + 1e-8), dim=1, keepdim=True)
                    aleatoric_uncertainty = entropy

                # 5. Backward Pass
                loss = loss.mean()
                epoch_train_loss += loss.item() * len(label)                
                global_steps += 1
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_postfix(train_loss=loss.item())

                # 6. Store Dynamics
                if epistemic_uncertainty is not None:
                    idx_list = torch.atleast_1d(batch["idx"]).tolist()
                    gp_list = torch.atleast_1d(gold_probs).tolist()
                    eu_list = torch.atleast_1d(epistemic_uncertainty.squeeze()).tolist()
                    au_list = torch.atleast_1d(aleatoric_uncertainty.squeeze()).tolist()
                    correct_flags = (preds == label).int().tolist()

                    for idx, gp, corr, eu, au in zip(idx_list, gp_list, correct_flags, eu_list, au_list):
                        # Ensure idx is valid key
                        if idx in train_dynamics:
                            train_dynamics[idx]["conf"].append(gp)
                            train_dynamics[idx]["corr"].append(corr)
                            train_dynamics[idx]["eu"].append(eu) 
                            train_dynamics[idx]["au"].append(au) 

        if args.wandb:
            wandb.log({"train/loss": loss.item(), "train/epoch": epoch}, step=global_steps)
                    
        train_loss = epoch_train_loss / len(train_loader.dataset)
        print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f}')
        
        # eval every epoch
        model.eval()
        with torch.no_grad():
            val_loss, val_metric_dict = eval_loop(args, model, plm_model, metrics_dict, val_loader, loss_fn, device)
            val_metric_score = val_metric_dict[args.monitor]
            val_metric_list.append(val_metric_score)
            val_loss_list.append(val_loss)
            
            if args.wandb:
                val_log = {"valid/loss": val_loss}
                for metric_name, metric_score in val_metric_dict.items():
                    val_log[f"valid/{metric_name}"] = metric_score
                wandb.log(val_log)
            print(f'EPOCH {epoch} VAL loss: {val_loss:.4f} {args.monitor}: {val_metric_score:.4f}')
    
        # early stopping
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
    
    # Save training dynamics
    with open("train_dynamics_edl_ce.pkl", "wb") as f:
        pickle.dump(train_dynamics, f)
    print("train_dynamics has been saved to train_dynamics.pkl")

# --- FIX 2: Corrected Eval Loop ---
def eval_loop(args, model, plm_model, metrics_dict, dataloader, loss_fn, device=None):
    total_loss = 0
    epoch_iterator = tqdm(dataloader)
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
            
        label = batch["label"]
        logits = model(plm_model, batch)

        # --- MAJOR FIX: Calculate Loss ONCE, OUTSIDE the metric loop ---
        # 1. Determine Preds and Loss based on problem type
        if args.problem_type == 'regression' and args.num_labels == 1:
            loss = loss_fn(logits.squeeze(), label.squeeze())
            preds = logits.squeeze()
            targets = label.squeeze()
        elif args.problem_type == 'multi_label_classification':
            loss = loss_fn(logits, label.float())
            preds = logits
            targets = label
        else:
            # Single Label Classification (Cross Entropy)
            loss = loss_fn(logits, label.long()) # Ensure label is long
            preds = torch.argmax(logits, 1)
            targets = label

        # 2. Update Metrics using the calculated preds
        for metric_name, metric in metrics_dict.items():
            metric(preds, targets)

        # 3. Accumulate Loss (Now safe because loss is definitely defined)
        total_loss += loss.item() * len(label)
        epoch_iterator.set_postfix(eval_loss=loss.item())
    
    metrics_result_dict = {}
    epoch_loss = total_loss / len(dataloader.dataset)
    for metric_name, metric in metrics_dict.items():
        metrics_result_dict[metric_name] = metric.compute().item()
        metric.reset()
    return epoch_loss, metrics_result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--pooling_method', type=str, default='attention1d', help='pooling method')
    parser.add_argument('--return_attentions', action='store_true', help='return attentions')
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help='pooling dropout')
    
    # dataset
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--dataset_config', type=str, default=None, help='config of dataset')
    parser.add_argument('--num_labels', type=int, default=None, help='number of labels')
    parser.add_argument('--problem_type', type=str, default=None, help='problem type')
    parser.add_argument('--pdb_type', type=str, default=None, choices=[None, 'AlphaFold2', 'ESMFold'], help='pdb type')
    
    # --- Single CSV Input ---
    parser.add_argument('--csv_file', type=str, default=None, help='Path to the single CSV file containing all data')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data used for training')
    
    parser.add_argument('--train_file', type=str, default=None, help='train file (JSON)')
    parser.add_argument('--valid_file', type=str, default=None, help='val file (JSON)')
    parser.add_argument('--test_file', type=str, default=None, help='test file (JSON)')
    parser.add_argument('--metrics', type=str, default=None, help='computation metrics')
    
    # train model
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--max_batch_token', type=int, default=10000, help='max number of token per batch')
    parser.add_argument('--max_train_epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--monitor', type=str, default=None, help='monitor metric')
    parser.add_argument('--structure_seqs', type=str, default=None, help='structure token')
    parser.add_argument('--loss_fn', type=str, default='cross_entropy', choices=['cross_entropy', 'focal_loss'], help='loss function')
    
    # save model
    parser.add_argument('--model_name', type=str, default=None, help='model name')
    parser.add_argument('--ckpt_root', default="ckpt", help='root directory to save trained models')
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    
    # wandb log
    parser.add_argument('--wandb', action='store_true', help='use wandb to log')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_project', type=str, default='SES-Adapter')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.structure_seqs is not None:
        args.structure_seqs = args.structure_seqs.split(',')
        
    # [NEW CODE - SAFE CONFIG LOADING]
    if args.dataset_config is not None:
        print(f"Loading configuration from {args.dataset_config}...")
        dataset_config = json.loads(open(args.dataset_config).read())
    else:
        print("No dataset_config provided. Relying on command-line arguments.")
        dataset_config = {}

    if args.dataset is None:
        args.dataset = dataset_config.get('dataset', 'custom_csv_dataset')
    if args.pdb_type is None:
        args.pdb_type = dataset_config.get('pdb_type', None)
        
    # Fallback to JSON config, but allow it to be empty
    if args.num_labels is None:
        args.num_labels = dataset_config.get('num_labels')
    if args.problem_type is None:
        args.problem_type = dataset_config.get('problem_type')
    if args.monitor is None:
        args.monitor = dataset_config.get('monitor', 'loss')
    
    metrics_dict = {}
    if args.metrics is None:
        args.metrics = dataset_config.get('metrics', 'accuracy')
    
    # Handle metrics string list
    if args.metrics != 'None':
        if isinstance(args.metrics, str):
            args.metrics = args.metrics.split(',')
        for m in args.metrics:
            if m == 'accuracy':
                if args.num_labels == 2: metrics_dict[m] = BinaryAccuracy()
                else: metrics_dict[m] = Accuracy(task="multiclass", num_classes=args.num_labels)
            elif m == 'recall':
                if args.num_labels == 2: metrics_dict[m] = BinaryRecall()
                else: metrics_dict[m] = Recall(task="multiclass", num_classes=args.num_labels)
            elif m == 'precision':
                if args.num_labels == 2: metrics_dict[m] = BinaryPrecision()
                else: metrics_dict[m] = Precision(task="multiclass", num_classes=args.num_labels)
            elif m == 'f1':
                if args.num_labels == 2: metrics_dict[m] = BinaryF1Score()
                else: metrics_dict[m] = F1Score(task="multiclass", num_classes=args.num_labels)
            elif m == 'mcc':
                if args.num_labels == 2: metrics_dict[m] = BinaryMatthewsCorrCoef()
                else: metrics_dict[m] = MatthewsCorrCoef(task="multiclass", num_classes=args.num_labels)
            elif m == 'auc':
                if args.num_labels == 2: metrics_dict[m] = BinaryAUROC()
                else: metrics_dict[m] = AUROC(task="multiclass", num_classes=args.num_labels)
            elif m == 'f1_max':
                metrics_dict[m] = MultilabelF1Max(num_labels=args.num_labels)
            elif m == 'spearman_corr':
                metrics_dict[m] = SpearmanCorrCoef()
            else:
                raise ValueError(f"Invalid metric: {m}")
        for metric_name, metric in metrics_dict.items():
            metric.to(device)            
        
    # create checkpoint directory
    if args.ckpt_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.ckpt_dir = os.path.join(args.ckpt_root, current_date)
    else:
        args.ckpt_dir = os.path.join(args.ckpt_root, args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # init wandb
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"Adapter-{args.dataset}"
        if args.model_name is None:
            args.model_name = f"{args.wandb_run_name}.pt"
        
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, 
            entity=args.wandb_entity, config=vars(args)
        )
    
    # build tokenizer and protein language model
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

    if args.structure_seqs is not None:
        if 'esm3_structure_seq' in args.structure_seqs: 
            args.vocab_size = max(plm_model.config.vocab_size, 4100)
        else:
            args.vocab_size = plm_model.config.vocab_size
    else:
        args.structure_seqs = []
    
    # load adapter model
    model = AdapterModel(args)
    model.to(device)

    def param_num(model):
        total = sum([param.numel() for param in model.parameters() if param.requires_grad])
        num_M = total/1e6
        if num_M >= 1000:
            return "Number of parameter: %.2fB" % (num_M/1e3)
        else:
            return "Number of parameter: %.2fM" % (num_M)
    print(param_num(model))
    
    def process_data_line(data):
        if args.problem_type == 'multi_label_classification':
            if isinstance(data['label'], str):
                label_list = data['label'].split(',')
                data['label'] = [int(l) for l in label_list]
                binary_list = [0] * args.num_labels
                for index in data['label']:
                    binary_list[index] = 1
                data['label'] = binary_list
            elif isinstance(data['label'], int):
                 pass 
        
        if args.max_seq_len is not None:
            data["aa_seq"] = data["aa_seq"][:args.max_seq_len]
            for seq in args.structure_seqs:
                if seq in data:
                    data[seq] = data[seq][:args.max_seq_len]
            token_num = min(len(data["aa_seq"]), args.max_seq_len)
        else:
            token_num = len(data["aa_seq"])
        return data, token_num
    
    # --- HELPER: Process List of Dicts ---
    def process_dataset_from_list(data_list):
        dataset, token_nums = [], []
        for i, l in enumerate(data_list):
            if 'idx' not in l: l['idx'] = i
            data, token_num = process_data_line(l)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums

    # --- CSV Processing Helper ---
    def process_dataset_from_csv(csv_path, train_ratio=0.8):
        print(f"Reading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        all_data = []
        for idx, row in df.iterrows():
            entry = {
                "idx": idx,
                "name": row['protein_name'],
                "label": int(row['label']),
                "aa_seq": row['aa_seq'],
                "cluster_id": row['cluster_id'] 
            }
            all_data.append(entry)
            
        print(f"Total samples loaded: {len(all_data)}")
        
        train_data, temp_data = train_test_split(
            all_data, train_size=train_ratio, random_state=args.seed, stratify=[d['label'] for d in all_data]
        )
        
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=args.seed, stratify=[d['label'] for d in temp_data]
        )
        
        train_ds, train_tokens = process_dataset_from_list(train_data)
        val_ds, val_tokens = process_dataset_from_list(val_data)
        test_ds, test_tokens = process_dataset_from_list(test_data)
        
        return (train_ds, train_tokens), (val_ds, val_tokens), (test_ds, test_tokens)

    # ---------------------------------------------------------
    # DATA LOADING LOGIC
    # ---------------------------------------------------------
    
    if args.csv_file is not None:
        print(f">>> Loading and splitting CSV file: {args.csv_file}")
        (train_dataset, train_token_num), (val_dataset, val_token_num), (test_dataset, test_token_num) = \
            process_dataset_from_csv(args.csv_file, args.train_ratio)

    elif args.train_file is not None and args.train_file.endswith('json'):
        def process_dataset_from_json(file):
            dataset, token_nums = [], []
            with open(file) as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    data, token_num = process_data_line(data)
                    data["idx"] = i 
                    dataset.append(data)
                    token_nums.append(token_num)
            return dataset, token_nums

        train_dataset, train_token_num = process_dataset_from_json(args.train_file)
        val_dataset, val_token_num = process_dataset_from_json(args.valid_file)
        test_dataset, test_token_num = process_dataset_from_json(args.test_file)
    else:
        if args.train_file == None:
            train_dataset, train_token_num = process_dataset_from_list(load_dataset(args.dataset)['train'])
            val_dataset, val_token_num = process_dataset_from_list(load_dataset(args.dataset)['validation'])
            test_dataset, test_token_num = process_dataset_from_list(load_dataset(args.dataset)['test'])

    if dataset_config.get('normalize') == 'min_max':
        train_dataset, val_dataset, test_dataset = min_max_normalize_dataset(train_dataset, val_dataset, test_dataset)
    
    print(">>> trainset: ", len(train_dataset))
    print(">>> valset: ", len(val_dataset))
    print(">>> testset: ", len(test_dataset))
    
    # --------------------------------------------------------------
    # IMPORTANT: Ensure your embedding functions (e_descriptor, etc)
    # are present here in your file! I have kept the Collate FN below.
    # --------------------------------------------------------------
    
    def collate_fn(examples):
        idx_list, aa_seqs, labels = [], [], []
        for e in examples:
            idx_list.append(e['idx'])
            aa_seq = e["aa_seq"]
            if 'prot_bert' in args.plm_model or "prot_t5" in args.plm_model:
                aa_seq = " ".join(list(aa_seq))
                aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
            elif 'ankh' in args.plm_model:
                aa_seq = list(aa_seq)
            aa_seqs.append(aa_seq)
            labels.append(e["label"])

        if 'ankh' in args.plm_model:
            aa_inputs = tokenizer.batch_encode_plus(aa_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        else:
            aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
    
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
    
        if args.problem_type == 'regression':
            labels = torch.as_tensor(labels, dtype=torch.float)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)

        data_dict = {
            "aa_input_ids": aa_input_ids, 
            "attention_mask": attention_mask, 
            "label": labels,
            "idx": torch.tensor(idx_list, dtype=torch.long)
        }
        return data_dict

    # metrics, optimizer, dataloader
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.problem_type == "single_label_classification":
        if args.loss_fn == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
        elif args.loss_fn == "focal_loss":
            train_labels = [e["label"] for e in train_dataset]
            alpha = []
            for i in range(args.num_labels):
                count = train_labels.count(i)
                alpha.append(len(train_labels) / count if count > 0 else 1.0)
            print(">>> alpha: ", alpha)
            loss_fn = MultiClassFocalLossWithAlpha(num_classes=args.num_labels, alpha=alpha, device=device)
    elif args.problem_type == "regression":
        loss_fn = nn.MSELoss()
    elif args.problem_type == "multi_label_classification":
        loss_fn = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, collate_fn=collate_fn, batch_sampler=BatchSampler(train_token_num, args.max_batch_token))
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, collate_fn=collate_fn, batch_sampler=BatchSampler(val_token_num, args.max_batch_token, False))
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, collate_fn=collate_fn, batch_sampler=BatchSampler(test_token_num, args.max_batch_token, False))
    
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)
    
    print("---------- Start Training ----------")
    train(args, model, plm_model, accelerator, metrics_dict, train_loader, val_loader, test_loader, loss_fn, optimizer, device)
    
    if args.wandb:
        wandb.finish()