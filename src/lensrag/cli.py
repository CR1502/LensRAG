"""Command line interface for dataset preparation, indexing, and retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import prepare
from .index import build_index, search_image, search_text, VectorIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lensrag")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Extract the pinned Flickr8k subset")
    prep.add_argument("--parquet", type=Path, help="Use an already downloaded shard")
    prep.add_argument("--data-dir", type=Path, default=Path("data"))
    prep.add_argument("--max-images", type=int, default=2000)
    prep.add_argument("--query-count", type=int, default=100)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--rebuild", action="store_true")

    idx = sub.add_parser("index", help="Build persistent CLIP cosine indexes")
    idx.add_argument("--data-dir", type=Path, default=Path("data"))
    idx.add_argument("--db-path", type=Path, default=Path("data/chroma"))
    idx.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    idx.add_argument("--batch-size", type=int, default=32)
    idx.add_argument("--rebuild", action="store_true")

    for name in ("text", "image"):
        search = sub.add_parser(name, help=f"Search with {name} input")
        search.add_argument("query", help="Text query or image path")
        search.add_argument("--target", choices=["images", "captions"],
                            default="images" if name == "text" else "captions")
        search.add_argument("-k", "--top-k", type=int, default=5)
        search.add_argument("--db-path", type=Path, default=Path("data/chroma"))
        search.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    stats = sub.add_parser("stats", help="Show prepared data and index counts")
    stats.add_argument("--data-dir", type=Path, default=Path("data"))
    stats.add_argument("--db-path", type=Path, default=Path("data/chroma"))

    evaluate = sub.add_parser("evaluate", help="Run held-out retrieval evaluation")
    evaluate.add_argument("--data-dir", type=Path, default=Path("data"))
    evaluate.add_argument("--db-path", type=Path, default=Path("data/chroma"))
    evaluate.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    evaluate.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    report = sub.add_parser("report", help="Generate evaluation report")
    report.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare(args.parquet, args.data_dir, args.max_images,
                           args.query_count, args.seed, args.rebuild)
        result = {key: manifest[key] for key in ("image_count", "caption_count",
                                                "indexed_caption_count", "digest")}
    elif args.command == "index":
        result = build_index(args.data_dir, args.db_path, device=args.device,
                             batch_size=args.batch_size, rebuild=args.rebuild)
    elif args.command == "text":
        result = search_text(args.query, args.target, args.top_k, args.db_path, args.device)
    elif args.command == "image":
        result = search_image(args.query, args.target, args.top_k, args.db_path, args.device)
    elif args.command == "stats":
        from .dataset import load_manifest
        manifest = load_manifest(args.data_dir)
        index = VectorIndex(args.db_path)
        result = {"prepared_images": manifest["image_count"],
                  "prepared_captions": manifest["caption_count"],
                  "heldout_queries": manifest["query_count"],
                  "indexed_images": index.images.count(),
                  "indexed_captions": index.captions.count(),
                  "manifest_digest": manifest["digest"]}
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(data_dir=args.data_dir, db_path=args.db_path,
                          output_dir=args.output_dir, device=args.device)
    else:
        from .evaluation import generate_report
        result = generate_report(output_dir=args.output_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
