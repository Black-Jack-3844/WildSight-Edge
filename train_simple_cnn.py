import argparse, os, random, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def list_images(root: str) -> Tuple[List[str], List[int], List[str]]:
    rootp = Path(root)
    classes = sorted([d.name for d in rootp.iterdir() if d.is_dir()])
    cls2idx = {c:i for i,c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        for p in (rootp / c).iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
                labels.append(cls2idx[c])
    return paths, labels, classes

def resize_shorter(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < h:
        nw = size
        nh = int(h * size / w)
    else:
        nh = size
        nw = int(w * size / h)
    return img.resize((nw, nh), Image.BILINEAR)

def center_crop(img: Image.Image, crop: int) -> Image.Image:
    w, h = img.size
    left = (w - crop) // 2
    top = (h - crop) // 2
    return img.crop((left, top, left + crop, top + crop))

def random_crop(img: Image.Image, crop: int) -> Image.Image:
    w, h = img.size
    if w == crop and h == crop:
        return img
    left = random.randint(0, max(0, w - crop))
    top = random.randint(0, max(0, h - crop))
    return img.crop((left, top, left + crop, top + crop))

def random_hflip(img: Image.Image, p: float = 0.5) -> Image.Image:
    if random.random() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def to_tensor_normalize(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

class SimpleFolderDataset(Dataset):
    def __init__(self, split_dir: str, train: bool):
        self.paths, self.labels, self.classes = list_images(split_dir)
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.labels[idx]
        img = Image.open(p)
        img = resize_shorter(img, 256)
        if self.train:
            img = random_crop(img, 224)
            img = random_hflip(img)
        else:
            img = center_crop(img, 224)
        x = to_tensor_normalize(img)
        return x, y

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_one_epoch(model, dl, device, crit, opt):
    model.train()
    run_loss = 0.0
    run_correct = 0
    n = 0
    for x, y in dl:
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        run_loss += loss.item() * x.size(0)
        run_correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return run_loss / n, run_correct / n

@torch.no_grad()
def eval_one_epoch(model, dl, device, crit):
    model.eval()
    run_loss = 0.0
    run_correct = 0
    n = 0
    for x, y in dl:
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        out = model(x)
        loss = crit(out, y)
        run_loss += loss.item() * x.size(0)
        run_correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return run_loss / n, run_correct / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)  # placeholder
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Missing train dir: {train_dir}")

    ds_train = SimpleFolderDataset(train_dir, train=True)
    classes = ds_train.classes
    ds_val = SimpleFolderDataset(val_dir, train=False) if os.path.isdir(val_dir) else None

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0) if ds_val else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "Classes:", classes, flush=True)

    model = SmallCNN(num_classes=len(classes)).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    os.makedirs(args.model_dir, exist_ok=True)
    best_acc = -1.0

    for ep in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, dl_train, device, crit, opt)
        msg = f"Epoch {ep+1}/{args.epochs} train loss {tr_loss:.4f} acc {tr_acc:.4f}"
        if dl_val:
            va_loss, va_acc = eval_one_epoch(model, dl_val, device, crit)
            msg += f" | val loss {va_loss:.4f} acc {va_acc:.4f}"
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"state_dict": model.state_dict(), "classes": classes}, os.path.join(args.model_dir, "resnet18.pth"))
        else:
            torch.save({"state_dict": model.state_dict(), "classes": classes}, os.path.join(args.model_dir, "resnet18.pth"))
        print(msg + f" | time {time.time()-t0:.1f}s", flush=True)

    print("Saved:", os.path.join(args.model_dir, "resnet18.pth"), flush=True)

if __name__ == "__main__":
    main()