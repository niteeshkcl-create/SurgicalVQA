import os
import sys
import argparse
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, average_precision_score

from src.data.dataset import CholecT50Dataset, get_transforms
from src.models.backbone import SurgicalVideoMAE

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for clips, labels in pbar:
        clips = clips.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        outputs = model(clips, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    all_preds = {'triplet': []}
    all_labels = {'triplet': []}
    
    for clips, labels in tqdm(val_loader, desc="Evaluating"):
        clips = clips.to(device)
        outputs = model(clips)
        logits = outputs['logits']
        
        for name in all_preds.keys():
            preds = torch.sigmoid(logits[name]).cpu().numpy()
            all_preds[name].extend(preds)
            all_labels[name].extend(labels[name].numpy())
            
    results = {}
    for name in all_preds.keys():
        preds_np = np.array(all_preds[name])
        labels_np = np.array(all_labels[name])
        
        # Calculate mean Average Precision (mAP) for multi-label tasks
        mAP = average_precision_score(labels_np, preds_np, average='macro')
        results[name] = mAP
        
    return results

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Official splits for CholecT50 (simplified for demo)
    # Fold 1 train: [79, 2, 51, 6, 25, 14, 66, 23, 50, 111]
    train_vids = [79, 2, 51, 6, 25, 14, 66, 23, 50, 111]
    val_vids = [80, 32, 5] # Subset for validation
    
    train_dataset = CholecT50Dataset(args.data_root, train_vids, transform=get_transforms('train'))
    val_dataset = CholecT50Dataset(args.data_root, val_vids, transform=get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    num_classes_dict = {'triplet': 100, 'phase': 100}
    model = SurgicalVideoMAE(num_classes_dict=num_classes_dict, use_flash_attn=args.use_flash_attn)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_mAP = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {loss:.4f}")
        
        results = evaluate(model, val_loader, device)
        print(f"Val Results: {results}")
        
        if results['triplet'] > best_mAP:
            best_mAP = results['triplet']
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"New best model saved with Triplet mAP: {best_mAP:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/processed/cholect50")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="models/checkpoints")
    parser.add_argument("--use_flash_attn", action="store_true")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
