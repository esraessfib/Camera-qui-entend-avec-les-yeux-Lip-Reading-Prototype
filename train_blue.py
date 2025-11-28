# train_from_metadata.py
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

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = r"F:\Documents\ensit\3ETA\traitement_image\projet\LipNet\data\prepared_blue\metadata_blue.csv"
MODEL_DIR = r"F:\Documents\ensit\3ETA\traitement_image\projet\LipNet\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Image size (height, width). Keep modest for GTX1650 (4GB).
IMG_H, IMG_W = 50, 100

# If you have more VRAM increase BATCH (e.g. 8)
DEFAULT_BATCH = 4

# Training params
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-5
CHECKPOINT_EVERY = 2  # epochs
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
    # Read metadata
    df = pd.read_csv(CSV_PATH)
    # keep only existing paths
    df = df[df['frames_path'].apply(lambda p: os.path.exists(p))].reset_index(drop=True)
    print(f"Dataset size after filter: {len(df)} samples")

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

    # Try torch.compile but gracefully fallback if fails (Windows/no Triton)
    try:
        if platform.system() != "Windows" and hasattr(torch, 'compile'):
            model = torch.compile(model)
            print("torch.compile enabled")
        else:
            print("torch.compile skipped (Windows or not available)")
    except Exception as e:
        print("torch.compile failed -> fallback to eager. Reason:", e)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []

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

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

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
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        scheduler.step(avg_val_loss)

        epoch_time = time.time() - t0
        print(f"\nEpoch {epoch} done in {epoch_time:.1f}s | Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc*100:.2f}%")

        # Save checkpoint
        if epoch % CHECKPOINT_EVERY == 0 or val_acc > best_val_acc:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accs': val_accs
            }
            ckpt_path = os.path.join(MODEL_DIR, f"lipnet_ckpt_e{epoch}.pth")
            torch.save(ckpt, ckpt_path)
            print("Checkpoint saved:", ckpt_path)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(MODEL_DIR, "lipnet_best.pth")
                torch.save(ckpt, best_path)
                print("New best model saved:", best_path)

        # show some prediction examples
        n_show = min(5, len(preds_all))
        print("Examples (True -> Pred):")
        for i in range(n_show):
            ok = "✓" if preds_all[i] == trues_all[i] else "✗"
            print(f"  {ok} '{trues_all[i]}' -> '{preds_all[i]}'")

    # end epochs
    print("\nTraining finished. Best val acc: {:.2f}%".format(best_val_acc*100))

    # Save final model (state_dict)
    final_path = os.path.join(MODEL_DIR, "lipnet_final.pth")
    torch.save({'model_state': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accs': val_accs}, final_path)
    print("Final model saved:", final_path)

    # Save a torchscript version for realtime (example input required)
    try:
        model.eval()
        example_frames = torch.zeros(1, min(12, train_ds[0][0].shape[0]), 1, IMG_H, IMG_W).to(device)  # (B, T, C, H, W)
        # Trace requires model to accept same input shape; wrap forward to accept (B,T,C,H,W)
        traced = torch.jit.trace(model, example_frames)
        ts_path = os.path.join(MODEL_DIR, "lipnet_realtime.pt")
        traced.save(ts_path)
        print("TorchScript model saved:", ts_path)
    except Exception as e:
        print("Could not save TorchScript model:", e)

    # Plots
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot([v*100 for v in val_accs], label='val_acc%')
    plt.legend(); plt.title('Val Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=200)
    plt.show()

if __name__ == "__main__":
    train()
