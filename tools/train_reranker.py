from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from weirwood_index.indexing import load_index
from weirwood_index.models import WeirwoodError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a local cross-encoder on mined narrative hard negatives."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise WeirwoodError("training parameters must be positive")
    if not args.base_model.is_dir():
        raise WeirwoodError(f"base model directory does not exist: {args.base_model}")
    payload = json.loads(args.training_data.read_text(encoding="utf-8"))
    if payload.get("purpose") != "narrative-reranker-hard-negatives":
        raise WeirwoodError("training data has the wrong purpose")
    if payload.get("benchmark", {}).get("split") == "acceptance":
        raise WeirwoodError("refusing to train on an acceptance benchmark")

    index = load_index(args.index)
    chunks = {chunk.id: chunk for chunk in index.chunks}
    samples: list[tuple[str, str, str]] = []
    for example in payload.get("examples", []):
        query = example["query"]
        positives = [chunks[chunk_id].text for chunk_id in example["positive_chunk_ids"]]
        negatives = [chunks[chunk_id].text for chunk_id in example["negative_chunk_ids"]]
        samples.extend(
            (query, positive, negative)
            for positive in positives
            for negative in negatives
        )
    if not samples:
        raise WeirwoodError("training data contains no examples")
    random.Random(args.seed).shuffle(samples)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        local_files_only=True,
    )

    def encode(queries: tuple[str, ...], passages: tuple[str, ...]) -> dict[str, Any]:
        return tokenizer(
            list(queries),
            list(passages),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    def collate(batch: list[tuple[str, str, str]]) -> dict[str, dict[str, Any]]:
        queries, positives, negatives = zip(*batch, strict=True)
        return {
            "positive": encode(queries, positives),
            "negative": encode(queries, negatives),
        }

    loader = DataLoader(
        samples,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.epochs * len(loader)
    warmup_steps = max(1, len(loader) // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup_steps)
        * max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            positive_scores = model(**batch["positive"]).logits.reshape(-1)
            negative_scores = model(**batch["negative"]).logits.reshape(-1)
            loss = torch.nn.functional.softplus(
                negative_scores - positive_scores
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach())
        print(
            f"epoch={epoch + 1} mean_loss={total_loss / len(loader):.6f}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metadata = {
        "training_data": str(args.training_data.resolve()),
        "index": str(args.index.resolve()),
        "base_model": str(args.base_model.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "samples": len(samples),
        "optimizer_steps": total_steps,
        "trainer": "torch-pairwise-softplus",
    }
    (args.output / "weirwood_training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
