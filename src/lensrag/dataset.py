"""Prepare a reproducible Flickr8k subset and held-out text queries."""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

DATASET_REPO = "tsystems/flickr8k"
DATASET_REVISION = "81fc5f3a41274c80f17b0406426d57cac57ce6fb"
SHARD_NAME = "data/train-00000-of-00003.parquet"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_manifest(data_dir: str | Path = "data") -> dict:
    path = Path(data_dir) / "manifest.json"
    manifest = json.loads(path.read_text())
    digest = manifest.pop("digest")
    if hashlib.sha256(_json_bytes(manifest)).hexdigest() != digest:
        raise ValueError(f"Manifest digest mismatch: {path}")
    manifest["digest"] = digest
    return manifest


def load_queries(data_dir: str | Path, manifest: dict) -> list[dict]:
    """Require the exact deterministic holdout, including order and caption text."""
    queries = json.loads((Path(data_dir) / "queries.json").read_text())
    sampled = random.Random(manifest["seed"]).sample(manifest["records"], manifest["query_count"])
    expected = [{"query_id": f"q-{i:04d}", "image_id": rec["image_id"],
                 "text": rec["captions"][0]} for i, rec in enumerate(sampled)]
    if queries != expected:
        raise ValueError("queries.json differs from the deterministic manifest holdout")
    return queries


def prepare(
    parquet_path: str | Path | None = None,
    data_dir: str | Path = "data",
    max_images: int = 2000,
    query_count: int = 100,
    seed: int = 42,
    rebuild: bool = False,
) -> dict:
    """Extract first distinct images in shard order and hold out query captions."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    queries_path = data_dir / "queries.json"
    if manifest_path.exists() and not rebuild:
        manifest = load_manifest(data_dir)
        if not queries_path.exists():
            raise ValueError("queries.json missing; rerun prepare with --rebuild")
        if (manifest["max_images"], manifest["query_count"], manifest["seed"]) != (
            max_images, query_count, seed
        ):
            raise ValueError("Prepared data uses different options; pass --rebuild")
        load_queries(data_dir, manifest)
        return manifest
    if max_images < 1 or not (0 <= query_count <= max_images):
        raise ValueError("Require max_images >= 1 and 0 <= query_count <= max_images")

    shard = Path(parquet_path) if parquet_path is not None else Path(
        hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=SHARD_NAME,
                        revision=DATASET_REVISION)
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    records: list[dict] = []
    seen: set[str] = set()
    parquet = pq.ParquetFile(shard)
    required = {"image", "image_filename", "captions"}
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError(f"Parquet is missing columns: {required - set(parquet.schema_arrow.names)}")
    for batch in parquet.iter_batches(batch_size=32, columns=list(required)):
        for row in batch.to_pylist():
            filename = str(row["image_filename"])
            if filename in seen:
                continue
            seen.add(filename)
            captions = [str(x).strip() for x in row["captions"] if x is not None and str(x).strip()]
            if not captions:
                raise ValueError(f"Image {filename} has no captions")
            raw = row["image"]
            if isinstance(raw, dict):
                raw = raw.get("bytes")
            if not raw:
                raise ValueError(f"Image {filename} has no embedded bytes")
            image_id = f"img-{len(records):06d}"
            relative_path = f"images/{image_id}.jpg"
            with Image.open(io.BytesIO(raw)) as image:
                image.convert("RGB").save(data_dir / relative_path, format="JPEG", quality=95)
            records.append({"image_id": image_id, "image_path": relative_path,
                            "image_filename": filename, "captions": captions})
            if len(records) == max_images:
                break
        if len(records) == max_images:
            break
    if len(records) < max_images:
        raise ValueError(f"Shard contains only {len(records)} distinct images; requested {max_images}")

    sampled = random.Random(seed).sample(records, query_count)
    queries = [{"query_id": f"q-{i:04d}", "image_id": rec["image_id"],
                "text": rec["captions"][0]} for i, rec in enumerate(sampled)]
    manifest = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "source_shard": SHARD_NAME,
        "source": str(shard.resolve()) if parquet_path is not None else f"hf://{DATASET_REPO}/{SHARD_NAME}",
        "max_images": max_images,
        "query_count": query_count,
        "seed": seed,
        "image_count": len(records),
        "caption_count": sum(len(r["captions"]) for r in records),
        "indexed_caption_count": sum(len(r["captions"]) for r in records) - query_count,
        "records": records,
    }
    manifest["digest"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    queries_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False) + "\n")
    return manifest
