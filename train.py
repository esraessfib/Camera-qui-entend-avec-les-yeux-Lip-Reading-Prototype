import os
import time
import math
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import Levenshtein
import numpy as np
from collections import Counter

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "..\projet\LipNet\data\prepared_blue\metadata_blue.csv"
MODEL_DIR = "..\projet\LipNet\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Image size (height, width). Keep modest for GTX1650 (4GB).
IMG_H, IMG_W = 50, 100

# If you have more VRAM increase BATCH (e.g. 8)
DEFAULT_BATCH = 4

# Training params - IMPROVED
EPOCHS = 100
LR = 1e-3    # Higher initial learning rate
WEIGHT_DECAY = 1e-5
CHECKPOINT_EVERY = 2  
NUM_WORKERS = 2

# -----------------------------
# VOCAB (characters) for CTC
# include blank '-' and the space character ' ' to separate words
# -----------------------------
VOCAB = ['-',' '] + [chr(i) for i in range(65, 91)]  # ['-', ' ', 'A'..'Z']
char2idx = {c:i for i,c in enumerate(VOCAB)}
idx2char = {i:c for c,i in char2idx.items()}

# -----------------------------
# Utilities
# -----------------------------
def normalize_text(s):
    # Keep letters and spaces only, uppercase
    s = str(s).upper()
    # replace any non-letter and non-space by nothing
    return ''.join(ch for ch in s if (ch == ' ' or ('A' <= ch <= 'Z'))).strip()

def text_to_indices(s):
    s = normalize_text(s)
    return [char2idx[c] for c in s if c in char2idx]

def decode_ctc_batch(output_tensor):
    # output_tensor: (B, T, C)
    pred = output_tensor.argmax(dim=2)  # (B, T)
    decoded = []
    for p in pred:
        prev = None
        chars = []
        for c in p.cpu().numpy():
            if c != 0 and c != prev:  # skip blank (index 0) and repeats
                chars.append(idx2char[c])
            prev = c
        decoded.append(''.join(chars).strip())
    return decoded

def word_accuracy(preds, trues):
    # average normalized similarity (1 - edit / max_len)
    accs = []
    for p, t in zip(preds, trues):
        if len(t) == 0:
            continue
        d = Levenshtein.distance(p, t)
        denom = max(len(p), len(t))
        acc = 1.0 - (d / denom) if denom > 0 else 0.0
        accs.append(max(0.0, acc))
    return float(np.mean(accs)) if accs else 0.0

def char_accuracy(preds, trues):
    """Calculate character-level accuracy"""
    total_chars = 0
    correct_chars = 0
    for p, t in zip(preds, trues):
        # Count matching characters at same positions
        for i in range(min(len(p), len(t))):
            total_chars += 1
            if p[i] == t[i]:
                correct_chars += 1
        total_chars += abs(len(p) - len(t))  # penalty for length mismatch
    return correct_chars / total_chars if total_chars > 0 else 0.0

# -----------------------------
# Dataset & collate
# -----------------------------
class FramesDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, max_frames=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.max_frames = max_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames_dir = row['frames_path']
        files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        if self.max_frames:
            files = files[:self.max_frames]
        imgs = []
        for f in files:
            p = os.path.join(frames_dir, f)
            img = Image.open(p).convert('L')  # grayscale to save memory
            if self.transform:
                img = self.transform(img)  # Tensor: (C=1,H,W)
            imgs.append(img)
        if len(imgs) == 0:
            raise ValueError(f"No frames in {frames_dir}")
        frames_tensor = torch.stack(imgs)  # (T, C, H, W)
        target_idx = torch.tensor(text_to_indices(row['text']), dtype=torch.long)
        return frames_tensor, target_idx

def pad_collate(batch):
    # batch: list of (T, C, H, W) and target tensors (L,)
    frames_list, targets = zip(*batch)
    T_max = max([f.shape[0] for f in frames_list])
    B = len(frames_list)
    C, H, W = frames_list[0].shape[1:]  # assume all same
    # allocate (B, T_max, C, H, W)
    padded = torch.zeros(B, T_max, C, H, W)
    for i, f in enumerate(frames_list):
        t = f.shape[0]
        padded[i, :t] = f
    # return as (B, T, C, H, W) and list of targets
    return padded, list(targets)

# -----------------------------
# Model (3D CNN + LSTM)
# -----------------------------
class LipModel(nn.Module):
    def __init__(self, num_classes, img_h=IMG_H, img_w=IMG_W, in_channels=1):
        super().__init__()
        # two conv3d blocks similar to before but with grayscale input
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 48, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
            nn.Dropout3d(0.1),

            nn.Conv3d(48, 96, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
            nn.Dropout3d(0.15)
        )
        # compute feature map spatial size after two (1,2,2) pools => H/4, W/4
        h_after = img_h // 4
        w_after = img_w // 4
        lstm_input = 96 * h_after * w_after
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=384, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(384*2, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W) expected by us
        # conv3d expects (B, C, T, H, W)
        x = x.permute(0,2,1,3,4).contiguous()
        x = self.conv3d(x)
        x = x.permute(0,2,1,3,4).contiguous()  # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B, T, C*H*W)
        x, _ = self.lstm(x)
        x = self.fc(x)  # (B, T, num_classes)
        return x

# -----------------------------
# Train / Eval Loop
# -----------------------------
def train():
    # Read metadata with better error handling
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows in CSV: {len(df)}")
    
    if len(df) == 0:
        raise ValueError("CSV file is empty!")
    
    if 'frames_path' not in df.columns:
        raise ValueError(f"'frames_path' column not found. Available columns: {df.columns.tolist()}")
    
    # Normalize paths and check existence
    df['frames_path'] = df['frames_path'].apply(lambda p: os.path.normpath(str(p)))
    exists_mask = df['frames_path'].apply(lambda p: os.path.exists(p) and os.path.isdir(p))
    
    print(f"Valid directories: {exists_mask.sum()}/{len(df)}")
    
    if exists_mask.sum() == 0:
        print("\n No valid frame directories found!")
        print("Example paths from CSV (first 3):")
        for path in df['frames_path'].head(3):
            print(f"  - {path} (exists: {os.path.exists(path)})")
        raise ValueError("No valid frame directories. Please check paths in CSV and ensure frames are extracted.")
    
    # Keep only existing paths
    df = df[exists_mask].reset_index(drop=True)
    print(f"Dataset size after filter: {len(df)} samples")
    
    # Dataset Statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    all_texts = df['text'].apply(normalize_text)
    print(f"Unique phrases: {all_texts.nunique()}")
    print(f"Average phrase length: {all_texts.str.len().mean():.1f} chars")
    print(f"Min length: {all_texts.str.len().min()}, Max length: {all_texts.str.len().max()}")
    print("\nMost common phrases:")
    print(all_texts.value_counts().head(10))
    print("="*60 + "\n")

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),            # gives (C=1,H,W) since input grayscale
    ])

    # detect GPU and set batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type=='cuda' else "CPU"
    # adapt batch size heuristically
    batch_size = DEFAULT_BATCH
    if device.type=='cuda':
        # small heuristic: GTX1650 ~4GB -> keep 4, otherwise set 8 for >6GB GPUs
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 8:
            batch_size = 8
        elif vram_gb >= 6:
            batch_size = 6
        else:
            batch_size = DEFAULT_BATCH

    train_ds = FramesDataset(train_df, transform=transform)
    val_ds = FramesDataset(val_df, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"Device: {device} ({gpu_name}), batch_size: {batch_size}, workers: {NUM_WORKERS}")

    # model
    model = LipModel(num_classes=len(VOCAB), img_h=IMG_H, img_w=IMG_W, in_channels=1).to(device)

    try:
        if platform.system() != "Windows" and hasattr(torch, 'compile'):
            model = torch.compile(model)
            print("torch.compile enabled")
        else:
            print("torch.compile skipped (Windows or not available)")
    except Exception as e:
        print("torch.compile failed -> fallback to eager. Reason:", e)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% warmup
    )

    best_val_acc = 0.0
    train_losses, val_losses, val_accs, val_char_accs = [], [], [], []

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for frames, targets in pbar:
            # frames: (B, T, C, H, W); targets: list of tensors
            frames = frames.to(device, non_blocking=True)
            outputs = model(frames)  # (B, T, C)
            outputs_log = outputs.log_softmax(2).permute(1,0,2)  # (T,B,C)

            targets_cat = torch.cat(targets).to(device)
            input_lengths = torch.full((frames.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long).to(device)

            loss = criterion(outputs_log, targets_cat, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # Step per batch for OneCycleLR

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        preds_all, trues_all = [], []
        with torch.no_grad():
            for frames, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]", leave=False):
                frames = frames.to(device, non_blocking=True)
                outputs = model(frames)
                outputs_log = outputs.log_softmax(2).permute(1,0,2)

                targets_cat = torch.cat(targets).to(device)
                input_lengths = torch.full((frames.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(device)
                target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long).to(device)

                vl = criterion(outputs_log, targets_cat, input_lengths, target_lengths).item()
                running_val_loss += vl

                preds = decode_ctc_batch(outputs)
                trues = ["".join([idx2char[i.item()] for i in t]) for t in targets]
                preds_all.extend(preds)
                trues_all.extend(trues)

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = word_accuracy(preds_all, trues_all)
        char_acc = char_accuracy(preds_all, trues_all)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        val_char_accs.append(char_acc)

        epoch_time = time.time() - t0
        
        # Enhanced logging
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{EPOCHS} SUMMARY")
        print(f"{'='*60}")
        print(f"Time: {epoch_time:.1f}s")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Word Accuracy: {val_acc*100:.2f}%")
        print(f"Val Char Accuracy: {char_acc*100:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        #  Prediction diversity analysis
        unique_preds = len(set(preds_all))
        print(f"\nPrediction Diversity: {unique_preds}/{len(preds_all)} unique ({unique_preds/len(preds_all)*100:.1f}%)")
        print("Top 5 most common predictions:")
        for pred, count in Counter(preds_all).most_common(5):
            print(f"  '{pred}': {count} times ({count/len(preds_all)*100:.1f}%)")

        # Save checkpoint
        if epoch % CHECKPOINT_EVERY == 0 or val_acc > best_val_acc:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'val_char_accs': val_char_accs
            }
            ckpt_path = os.path.join(MODEL_DIR, f"lipnet_ckpt_e{epoch}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"\n✓ Checkpoint saved: {ckpt_path}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(MODEL_DIR, "lipnet_best.pth")
                torch.save(ckpt, best_path)
                print(f"✓ New best model saved: {best_path} (Val Acc: {val_acc*100:.2f}%)")

        # show some prediction examples
        print("\nPrediction Examples (True -> Pred):")
        n_show = min(8, len(preds_all))
        indices = np.random.choice(len(preds_all), n_show, replace=False)
        for i in indices:
            ok = "✓" if preds_all[i] == trues_all[i] else "✗"
            print(f"  {ok} '{trues_all[i]}' -> '{preds_all[i]}'")
        print(f"{'='*60}\n")

    # end epochs
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Final validation accuracy: {val_acc*100:.2f}%")
    print(f"Final char accuracy: {char_acc*100:.2f}%")

    # Save final model (state_dict)
    final_path = os.path.join(MODEL_DIR, "lipnet_final.pth")
    torch.save({'model_state': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'val_char_accs': val_char_accs}, final_path)
    print(f"\n✓ Final model saved: {final_path}")

    # Save a torchscript version for realtime 
    try:
        model.eval()
        example_frames = torch.zeros(1, min(12, train_ds[0][0].shape[0]), 1, IMG_H, IMG_W).to(device)
        traced = torch.jit.trace(model, example_frames)
        ts_path = os.path.join(MODEL_DIR, "lipnet_realtime.pt")
        traced.save(ts_path)
        print(f"✓ TorchScript model saved: {ts_path}")
    except Exception as e:
        print(f"Could not save TorchScript model: {e}")

    # Enhanced Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Word accuracy
    axes[0, 1].plot([v*100 for v in val_accs], label='Word Accuracy', linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Validation Word Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Character accuracy
    axes[1, 0].plot([v*100 for v in val_char_accs], label='Char Accuracy', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Validation Character Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined view
    axes[1, 1].plot([v*100 for v in val_accs], label='Word Acc', linewidth=2, color='green')
    axes[1, 1].plot([v*100 for v in val_char_accs], label='Char Acc', linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Word vs Character Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=200)
    print(f"✓ Training curves saved: {plot_path}")
    plt.show()
    
    print("="*60)

if __name__ == "__main__":
    train()