import argparse
import os
from pathlib import Path
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms

def get_dataloaders(data_dir, batch_size=8, workers=0):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
        if os.path.isdir(os.path.join(data_dir, x))
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=workers, pin_memory=False)
        for x in image_datasets
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets['train'].classes if 'train' in image_datasets else []
    return dataloaders, dataset_sizes, class_names

def build_model(num_classes, use_pretrained=True):
    # 预训练权重可能需要联网下载；若失败则自动退回不使用预训练
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
    try:
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_model(model, dataloaders, dataset_sizes, device, epochs=1, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase not in dataloaders:
                continue

            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            start = time.time()
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]
            elapsed = time.time() - start
            print(f"{phase}: loss {epoch_loss:.4f} acc {epoch_acc:.4f} time {elapsed:.1f}s")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    print(f"Best val acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to dataset root containing train/val/test")
    ap.add_argument("--model_dir", required=True, help="Where to save trained weights")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true", help="Do not use ImageNet pretrained weights")
    args = ap.parse_args()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    dataloaders, dataset_sizes, class_names = get_dataloaders(args.data_dir, args.batch_size, args.workers)
    if 'train' not in dataloaders:
        raise RuntimeError("No 'train' split found. Ensure your dataset has Data/train/<class> folders.")
    num_classes = len(class_names)
    print("Classes:", class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model(num_classes, use_pretrained=not args.no_pretrained).to(device)
    model = train_model(model, dataloaders, dataset_sizes, device, epochs=args.epochs)

    # Save weights and classes
    weights_path = os.path.join(args.model_dir, "resnet18.pth")
    torch.save({"state_dict": model.state_dict(), "classes": class_names}, weights_path)
    print("Saved:", weights_path)

if __name__ == "__main__":
    main()