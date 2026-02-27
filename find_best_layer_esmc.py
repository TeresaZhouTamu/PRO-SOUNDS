import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import os
from typing import List
from utils.esmc_utils import get_esmc_layer_and_feature_dim

class LinearProbe(nn.Module):
    """A Logistic Regression model for binary classification."""
    def __init__(self, input_dim):
        super().__init__()
        # nn.Linear implements the core linear transformation: W*x + b
        self.linear = nn.Linear(input_dim, 1) 

    def forward(self, x):
        return self.linear(x)

def train_linear_probe(features: torch.Tensor, labels: torch.Tensor, device: torch.device, 
                       epochs: int = 20, batch_size: int = 64, split_ratio: float = 0.8) -> float:
    """
    Trains a Linear Probe on activation features and returns validation accuracy.
    Includes class weighting to handle imbalanced datasets.
    """
    
    N = features.size(0)
    perm = torch.randperm(N)
    train_size = int(N * split_ratio)
    
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    X_train, y_train = features[train_indices].to(device), labels[train_indices].to(device)
    X_val, y_val = features[val_indices].to(device), labels[val_indices].to(device)
    
    if len(val_indices) == 0:
        X_val, y_val = X_train, y_train

    num_pos = (y_train == 1).sum().item()
    num_neg = (y_train == 0).sum().item()
    
    pos_weight_value = (num_neg / num_pos) if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value]).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = features.size(1)
    model = LinearProbe(input_dim).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val).squeeze(1)
        preds = (torch.sigmoid(val_outputs) > 0.5).long()

        correct = (preds == y_val).sum().item()
        accuracy = correct / len(y_val)
        
    return accuracy


def find_optimal_layer(pos_features_list: list[torch.Tensor], 
                       neg_features_list: list[torch.Tensor], 
                       n_layers: int, 
                       device: torch.device) -> int:
    """
    Finds the layer with the highest linear separation accuracy.
    """
    
    best_accuracy = -1.0
    best_layer = -1
    
    if not pos_features_list or not neg_features_list:
        print("Error: Feature lists are empty.")
        return 0

    n_pos = pos_features_list[0].size(0)
    n_neg = neg_features_list[0].size(0)
    
    labels = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])
    
    print(f"\nTraining on {n_pos} Positive and {n_neg} Negative samples...")
    print("--- Starting Linear Probing ---")
    
    for i in range(n_layers):
        if i >= len(pos_features_list) or i >= len(neg_features_list):
             continue

        # Combine features for current layer
        # pos_features_list[i] shape: (N_pos, hidden_dim)
        # neg_features_list[i] shape: (N_neg, hidden_dim)
        layer_features = torch.cat([pos_features_list[i], neg_features_list[i]], dim=0)
        
        # Train
        # Move to CPU first to save GPU memory between iterations if needed, 
        # but function moves them back to device.
        accuracy = train_linear_probe(layer_features, labels, device)
        
        print(f"Layer {i}: Accuracy = {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_layer = i
            
    print("---------------------------------------------------------")
    print(f"Optimal Layer: **Layer {best_layer}** (Accuracy: {best_accuracy:.4f})")
    
    return best_layer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the optimal ESMC layer using Linear Probing.")
    
    parser.add_argument('--pos_features_path', type=str, required=True, 
                        help="Path to .pt file with Positive features (from feature_extraction).")
    parser.add_argument('--neg_features_path', type=str, required=True, 
                        help="Path to .pt file with Negative features (from feature_extraction).")
    parser.add_argument('--model', type=str, default="600M", help="ESMC model size.")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use.")

    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading positive features: {args.pos_features_path}")
    pos_features_list: List[torch.Tensor] = torch.load(args.pos_features_path)
    print(f"Loading negative features: {args.neg_features_path}")
    neg_features_list: List[torch.Tensor] = torch.load(args.neg_features_path)

    n_layers, _ = get_esmc_layer_and_feature_dim()
    n_layers = min(len(pos_features_list), len(neg_features_list), n_layers)
    optimal_layer = find_optimal_layer(pos_features_list, neg_features_list, n_layers, device)
    
    print("\n---------------------------------------------------------")
    print(f"The optimal layer is: {optimal_layer}")
    print("---------------------------------------------------------")