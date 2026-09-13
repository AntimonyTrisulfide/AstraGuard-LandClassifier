"""Train DeepLabV3+ or SegFormer on prepared AstraGuard tiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from .classes import CLASS_NAMES, IGNORE_INDEX
from .config import load_config, with_runtime_overrides
from .dataset import H5LandCoverDataset, load_stats
from .engine import train_one_epoch, validate
from .losses import CombinedSegmentationLoss
from .models import build_model
from .utils import select_device, set_seed, write_json


def _make_loader(
    dataset: H5LandCoverDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
    )


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _append_history(path: Path, row: dict[str, Any]) -> None:
    flattened = {
        "epoch": row["epoch"],
        "train_loss": row["train_loss"],
        "val_loss": row["validation"]["loss"],
        "val_miou": row["validation"]["miou"],
        "val_macro_f1": row["validation"]["macro_f1"],
        "val_agriculture_iou": row["validation"]["per_class"]["agriculture"]["iou"],
        "val_agriculture_f1": row["validation"]["per_class"]["agriculture"]["f1"],
        "val_agriculture_area_error_percent": row["validation"][
            "agriculture_area_error_percent"
        ],
    }
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened))
        if not exists:
            writer.writeheader()
        writer.writerow(flattened)


def run_training(config: dict[str, Any], resume: Path | None = None) -> Path:
    training = config["training"]
    data_config = config["data"]
    seed = int(training.get("seed", 42))
    set_seed(seed)
    device = select_device()
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    processed_dir = Path(data_config["processed_dir"])
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = load_stats(processed_dir)
    scale_divisor = float(data_config.get("scale_divisor", stats["scale_divisor"]))

    train_dataset = H5LandCoverDataset(
        processed_dir / "train.h5",
        band_mean=stats["band_mean"],
        band_std=stats["band_std"],
        scale_divisor=scale_divisor,
        augment=bool(data_config.get("augment", True)),
    )
    validation_dataset = H5LandCoverDataset(
        processed_dir / "val.h5",
        band_mean=stats["band_mean"],
        band_std=stats["band_std"],
        scale_divisor=scale_divisor,
        augment=False,
    )
    batch_size = int(training.get("batch_size", 16))
    workers = int(data_config.get("num_workers", 4))
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=True,
        device=device,
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        device=device,
    )

    model = build_model(
        config["model"], initialize_pretrained=resume is None
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 60))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device)
    criterion = CombinedSegmentationLoss(
        num_classes=len(CLASS_NAMES),
        ignore_index=IGNORE_INDEX,
        class_weights=class_weights,
        ce_weight=float(training.get("ce_weight", 0.6)),
        dice_weight=float(training.get("dice_weight", 0.4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_miou = -1.0
    stale_epochs = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_miou", -1.0))

    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    write_json(output_dir / "data_stats.json", stats)

    patience = int(training.get("early_stopping_patience", 10))
    print(
        json.dumps(
            {
                "device": str(device),
                "train_tiles": len(train_dataset),
                "validation_tiles": len(validation_dataset),
                "amp": use_amp,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp=use_amp,
        )
        validation = validate(
            model,
            validation_loader,
            criterion,
            device,
            num_classes=len(CLASS_NAMES),
            ignore_index=IGNORE_INDEX,
            use_amp=use_amp,
        )
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "validation": validation}
        _append_history(output_dir / "history.csv", row)
        write_json(output_dir / "latest_metrics.json", row)
        current_miou = float(validation["miou"] or 0.0)
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou": max(best_miou, current_miou),
            "config": config,
            "stats": stats,
            "class_names": list(CLASS_NAMES),
        }
        _save_checkpoint(output_dir / "last.pt", payload)
        if current_miou > best_miou:
            best_miou = current_miou
            stale_epochs = 0
            payload["best_miou"] = best_miou
            _save_checkpoint(output_dir / "best.pt", payload)
        else:
            stale_epochs += 1

        agriculture = validation["per_class"]["agriculture"]
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} miou={current_miou:.4f} "
            f"agri_iou={agriculture['iou']} agri_f1={agriculture['f1']}"
        )
        if stale_epochs >= patience:
            print(f"Early stopping after {stale_epochs} epochs without mIoU improvement")
            break
    return output_dir / "best.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", help="Override data.processed_dir")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = with_runtime_overrides(
        load_config(args.config), data_dir=args.data_dir, output_dir=args.output_dir
    )
    checkpoint = run_training(config, args.resume)
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
