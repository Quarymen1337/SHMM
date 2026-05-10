from __future__ import annotations

import argparse
import json
from pathlib import Path

from .make_neat_to_bnn import run_make_neat_to_bnn
from .train_gan import run_train_gan, finetune_generator
from .train_neat import run_train_neat
from .train_workability import run_train_workability


def _write_payload(payload: str, output_path: str | None) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialgen",
        description="Concrete mix design: inverse NEAT training and BNN fine-tuning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    neat_parser = subparsers.add_parser(
        "train_neat",
        help="Train the inverse NEAT network and save it under artifacts/train_neat",
    )
    neat_parser.add_argument("--config", required=True, help="Path to backward.json")
    neat_parser.add_argument("--artifacts-dir", default="artifacts", help="Root directory for all stage artifacts")
    neat_parser.add_argument("--inverse-dir", default=None, help="Optional override for the train_neat artifacts folder")
    neat_parser.add_argument("--output", default=None, help="Optional path for JSON summary")

    bnn_parser = subparsers.add_parser(
        "make_neat_to_bnn",
        help="Convert trained NEAT network into a Bayesian NN and fine-tune on known data",
    )
    bnn_parser.add_argument("--config", required=True, help="Path to make_neat_to_bnn.json")
    bnn_parser.add_argument("--artifacts-dir", default="artifacts", help="Root directory for all stage artifacts")
    bnn_parser.add_argument("--inverse-dir", default=None, help="Optional override for the train_neat artifacts folder")
    bnn_parser.add_argument("--bnn-dir", default=None, help="Optional override for the make_neat_to_bnn artifacts folder")
    bnn_parser.add_argument("--output", default=None, help="Optional path for JSON summary")

    gan_parser = subparsers.add_parser(
        "train_gan",
        help="Train conditional GAN for strength prediction with a NEAT+BNN discriminator",
    )
    gan_parser.add_argument("--config", required=True, help="Path to gan.json")
    gan_parser.add_argument("--artifacts-dir", default="artifacts", help="Root directory for all stage artifacts")
    gan_parser.add_argument("--gan-dir", default=None, help="Optional override for the train_gan artifacts folder")
    gan_parser.add_argument("--output", default=None, help="Optional path for JSON summary")

    work_parser = subparsers.add_parser(
        "train_workability",
        help="Train MC-Dropout MLP to predict workability (slump) from mix composition",
    )
    work_parser.add_argument("--config", required=True, help="Path to workability.json")
    work_parser.add_argument("--artifacts-dir", default="artifacts")
    work_parser.add_argument("--output-dir", default=None)
    work_parser.add_argument("--output", default=None)

    ft_parser = subparsers.add_parser(
        "finetune_gan",
        help="Fine-tune pre-trained GAN generator on a narrow lab dataset",
    )
    ft_parser.add_argument("--pretrained", required=True, help="Path to generator.pt checkpoint")
    ft_parser.add_argument("--data", required=True, help="Path to narrow lab CSV")
    ft_parser.add_argument("--output-dir", default="artifacts/finetune")
    ft_parser.add_argument("--epochs", type=int, default=60)
    ft_parser.add_argument("--lr", type=float, default=2e-4)
    ft_parser.add_argument("--no-freeze", action="store_true", help="Fine-tune all layers")
    ft_parser.add_argument("--output", default=None)

    return parser


def _handle_train_neat(args) -> int:
    summary = run_train_neat(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        inverse_dir=args.inverse_dir,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_make_neat_to_bnn(args) -> int:
    summary = run_make_neat_to_bnn(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        inverse_dir=args.inverse_dir,
        bnn_dir=args.bnn_dir,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_train_gan(args) -> int:
    summary = run_train_gan(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        gan_dir=args.gan_dir,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_train_workability(args) -> int:
    summary = run_train_workability(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_finetune_gan(args) -> int:
    summary = finetune_generator(
        pretrained_path=args.pretrained,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        freeze_encoder=not args.no_freeze,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "train_neat": _handle_train_neat,
        "make_neat_to_bnn": _handle_make_neat_to_bnn,
        "train_gan": _handle_train_gan,
        "train_workability": _handle_train_workability,
        "finetune_gan": _handle_finetune_gan,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2
    return handler(args)
