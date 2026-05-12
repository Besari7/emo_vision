from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from multimodal_emotion.evaluation.metrics import classification_metrics, confusion_records


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device: torch.device, labels: list[str]) -> tuple[dict, list[dict]]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_true: list[int] = []
    all_pred: list[int] = []
    all_probs: list[np.ndarray] = []
    prediction_rows: list[dict] = []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            audio_features=batch["audio_features"],
            video_features=batch["video_features"],
            audio_mask=batch["audio_mask"],
            video_mask=batch["video_mask"],
        )
        logits = outputs["logits"]
        loss = loss_fn(logits, batch["labels"])
        probabilities = outputs["probabilities"]
        predictions = torch.argmax(logits, dim=1)

        total_loss += float(loss.item()) * len(batch["labels"])
        all_true.extend(batch["labels"].cpu().tolist())
        all_pred.extend(predictions.cpu().tolist())
        all_probs.extend(probabilities.cpu().numpy())

        for sample_id, prediction, probability in zip(
            batch["sample_ids"],
            predictions.cpu().tolist(),
            probabilities.cpu().numpy(),
            strict=True,
        ):
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "predicted_label": labels[prediction],
                    "probabilities": {label: float(score) for label, score in zip(labels, probability, strict=True)},
                }
            )

    average_loss = total_loss / max(len(dataloader.dataset), 1)
    probabilities_array = np.asarray(all_probs)
    metrics = classification_metrics(all_true, all_pred, labels, probabilities_array)
    metrics["loss"] = average_loss
    metrics["confusion_matrix"] = confusion_records(all_true, all_pred, labels)
    return metrics, prediction_rows


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    config,
    output_dir: str | Path,
) -> dict:
    device = resolve_device(config.training.device)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_metric = float("-inf")
    best_epoch = 0
    history: list[dict] = []
    patience_counter = 0

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_dataloader:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_features=batch["audio_features"],
                video_features=batch["video_features"],
                audio_mask=batch["audio_mask"],
                video_mask=batch["video_mask"],
            )
            loss = loss_fn(outputs["logits"], batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["labels"])

        train_loss = total_loss / max(len(train_dataloader.dataset), 1)
        val_metrics, _ = evaluate_model(model, val_dataloader, device, config.model.labels)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)

        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_metric,
                },
                output_root / "best_model.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= config.training.early_stopping_patience:
            break

    summary = {
        "best_epoch": best_epoch,
        "best_macro_f1": best_metric,
        "history": history,
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
