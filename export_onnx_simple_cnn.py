import argparse, os
import torch
import torch.nn as nn

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
    ap.add_argument("--model_dir", required=True, help="Folder containing resnet18.pth")
    ap.add_argument("--num_classes", type=int, required=True)
    args = ap.parse_args()

    ckpt = os.path.join(args.model_dir, "resnet18.pth")
    onnx_path = os.path.join(args.model_dir, "resnet18.onnx")
    state = torch.load(ckpt, map_location="cpu")
    model = SmallCNN(args.num_classes)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input_0"], output_names=["output_0"],
        opset_version=18,  # 使用 18，避免降级时的转换报错
        do_constant_folding=True
    )
    print("Exported ONNX to:", onnx_path)

if __name__ == "__main__":
    main()