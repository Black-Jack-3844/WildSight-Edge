import argparse, os
import cv2
import numpy as np
import onnx
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_labels(p):
    with open(p, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def preprocess_bgr(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = 256.0 / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (nh - 224) // 2
    left = (nw - 224) // 2
    crop = resized[top:top+224, left:left+224]
    x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,224,224)
    return x, crop

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="output_onnx.jpg")
    args = ap.parse_args()

    # 校验 ONNX
    onnx_model = onnx.load(args.model)
    onnx.checker.check_model(onnx_model)

    labels = load_labels(args.labels)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    img = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.img)
    x, vis = preprocess_bgr(img)

    y = sess.run([out_name], {in_name: x})[0]  # (1, C)
    probs = softmax(y)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    label = labels[idx] if 0 <= idx < len(labels) else str(idx)
    text = f"{label} {conf*100:.2f}%"
    cv2.rectangle(vis, (0,0), (vis.shape[1], 30), (0,0,0), -1)
    cv2.putText(vis, text, (5,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(args.out, vis)
    print("Prediction:", text)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()