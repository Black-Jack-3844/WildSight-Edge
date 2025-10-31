import argparse, os
import cv2
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def preprocess_bgr(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = 256.0 / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (nh - 224) // 2
    left = (nw - 224) // 2
    cropped = resized[top:top+224, left:left+224]
    img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor, cropped

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="output.jpg")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    num_classes = len(labels)

    ckpt = os.path.join(args.model_dir, "resnet18.pth")
    state = torch.load(ckpt, map_location="cpu")
    model = SmallCNN(num_classes)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()

    img_bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.img}")

    x, vis = preprocess_bgr(img_bgr)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

    label = labels[idx] if 0 <= idx < len(labels) else str(idx)
    text = f"{label} {conf*100:.2f}%"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), thickness=-1)
    cv2.putText(vis, text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(args.out, vis)
    print(f"Prediction: {text}")
    print(f"Saved prediction image to: {args.out}")

if __name__ == "__main__":
    main()