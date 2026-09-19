"""Train the 4-state traffic-light crop classifier and export safe NPZ weights."""

import argparse
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modules.provenance import file_record
from modules.traffic_light_classifier import build_model, CLASSES


class CropDataset(Dataset):
    def __init__(self, root, augment=False):
        self.samples, self.augment = [], bool(augment)
        root = Path(root)
        for class_id, name in enumerate(CLASSES):
            folder = root / name
            if folder.is_dir():
                self.samples.extend((path, class_id) for path in folder.iterdir()
                                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read {path}")
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        if self.augment and random.random() < 0.5:
            image = cv2.convertScaleAbs(image, alpha=random.uniform(0.75, 1.25),
                                        beta=random.uniform(-12, 12))
        image = image[:, :, ::-1].copy()
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, label


def evaluate(model, loader, device):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu().numpy()
            for truth, prediction in zip(labels.numpy(), predictions):
                matrix[int(truth), int(prediction)] += 1
    f1s = []
    for index in range(len(CLASSES)):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1s)), matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="root containing train/{red,yellow,green,unknown} and val/...")
    parser.add_argument("--output", default="weights/traffic_light_state.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = Path(args.dataset)
    train_set, val_set = CropDataset(root / "train", True), CropDataset(root / "val", False)
    if len(train_set) == 0 or len(val_set) == 0:
        raise SystemExit("traffic-light dataset is empty or missing train/val class folders")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=2)
    best_f1, best_state, best_matrix, stale = -1.0, None, None, 0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(images), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        macro_f1, matrix = evaluate(model, val_loader, device)
        print(f"epoch={epoch + 1} macro_f1={macro_f1:.4f}")
        if macro_f1 > best_f1 + 1e-5:
            best_f1, stale = macro_f1, 0
            best_matrix = matrix.copy()
            best_state = {name: tensor.detach().cpu().numpy().copy()
                          for name, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **best_state)
    report = {"macro_f1": best_f1, "target_macro_f1": 0.90,
              "acceptance_pass": best_f1 >= 0.90,
              "confusion_matrix": best_matrix.tolist(), "classes": CLASSES,
              "artifact": file_record(output), "settings": vars(args)}
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
