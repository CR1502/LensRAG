"""Persistent cosine indexes for Flickr8k images and captions."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from .clip import CLIPEmbedder, MODEL_ID, MODEL_REVISION
from .dataset import load_manifest, load_queries

IMAGE_COLLECTION = "lensrag_images"
CAPTION_COLLECTION = "lensrag_captions"


class VectorIndex:
    def __init__(self, db_path: str | Path = "data/chroma"):
        self.client = chromadb.PersistentClient(path=str(Path(db_path)))
        self.images = self.client.get_or_create_collection(
            IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.captions = self.client.get_or_create_collection(
            CAPTION_COLLECTION, metadata={"hnsw:space": "cosine"})

    def clear(self) -> None:
        for name in (IMAGE_COLLECTION, CAPTION_COLLECTION):
            self.client.delete_collection(name)
        self.images = self.client.get_or_create_collection(
            IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.captions = self.client.get_or_create_collection(
            CAPTION_COLLECTION, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def query(collection, vector, n_results: int = 5) -> list[dict]:
        if n_results < 1:
            raise ValueError("n_results must be positive")
        if collection.count() == 0:
            return []
        result = collection.query(query_embeddings=[vector.tolist()],
                                  n_results=min(n_results, collection.count()),
                                  include=["metadatas", "distances", "documents"])
        rows = []
        for identifier, meta, document, distance in zip(
            result["ids"][0], result["metadatas"][0],
            result["documents"][0], result["distances"][0]
        ):
            rows.append({"id": identifier, "image_id": meta["image_id"],
                         "image_path": meta["image_path"],
                         "caption": document if collection.name == CAPTION_COLLECTION else None,
                         "distance": float(distance)})
        return rows


def _expected(manifest: dict, queries: list[dict]) -> tuple[list[dict], list[tuple[str, dict, str]]]:
    heldout = {query["image_id"] for query in queries}
    records = manifest["records"]
    captions = []
    for rec in records:
        for i, caption in enumerate(rec["captions"]):
            if i == 0 and rec["image_id"] in heldout:
                continue
            captions.append((f"{rec['image_id']}::caption-{i}", rec, caption))
    return records, captions


def build_index(data_dir: str | Path = "data", db_path: str | Path = "data/chroma",
                model_id: str = MODEL_ID, model_revision: str = MODEL_REVISION,
                device: str = "cpu", batch_size: int = 32,
                rebuild: bool = False) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)
    queries = load_queries(data_dir, manifest)
    records, captions = _expected(manifest, queries)
    index = VectorIndex(db_path)
    if rebuild:
        index.clear()
    metadata = {"manifest_digest": manifest["digest"],
                "model_id": model_id, "model_revision": model_revision}
    for collection in (index.images, index.captions):
        old = collection.metadata or {}
        if collection.count() and any(old.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"{collection.name} metadata differs; pass --rebuild")
        collection.modify(metadata=metadata)

    image_ids = {r["image_id"] for r in records}
    caption_ids = {item[0] for item in captions}
    existing_images = set(index.images.get(include=[])["ids"])
    existing_captions = set(index.captions.get(include=[])["ids"])
    if existing_images - image_ids or existing_captions - caption_ids:
        raise ValueError("Index has stale records; pass --rebuild")
    if existing_images != image_ids or existing_captions != caption_ids:
        embedder = CLIPEmbedder(model_id=model_id, revision=model_revision,
                                device=device, batch_size=batch_size)
        for start in range(0, len(records), batch_size):
            batch = [r for r in records[start:start + batch_size]
                     if r["image_id"] not in existing_images]
            if not batch:
                continue
            vectors = embedder.images([data_dir / r["image_path"] for r in batch])
            index.images.add(ids=[r["image_id"] for r in batch], embeddings=vectors.tolist(),
                             metadatas=[{"image_id": r["image_id"],
                                         "image_path": r["image_path"]} for r in batch],
                             documents=[r["image_filename"] for r in batch])
        for start in range(0, len(captions), batch_size):
            batch = [item for item in captions[start:start + batch_size]
                     if item[0] not in existing_captions]
            if not batch:
                continue
            vectors = embedder.texts([item[2] for item in batch])
            index.captions.add(ids=[item[0] for item in batch], embeddings=vectors.tolist(),
                               metadatas=[{"image_id": item[1]["image_id"],
                                           "image_path": item[1]["image_path"]} for item in batch],
                               documents=[item[2] for item in batch])
    return {"images": index.images.count(), "captions": index.captions.count(),
            "manifest_digest": manifest["digest"], "model_id": model_id,
            "model_revision": model_revision}


def _search(vector, target: str, db_path: str | Path, n_results: int) -> list[dict]:
    index = VectorIndex(db_path)
    if target not in {"images", "captions"}:
        raise ValueError("target must be images or captions")
    collection = index.images if target == "images" else index.captions
    return index.query(collection, vector, n_results=n_results)


def _model_for(db_path: str | Path, target: str) -> tuple[str, str]:
    index = VectorIndex(db_path)
    collection = index.images if target == "images" else index.captions
    if not collection.count():
        raise ValueError(f"{target} index is empty; run lensrag index")
    meta = collection.metadata or {}
    return meta["model_id"], meta["model_revision"]


def search_text(query: str, target: str = "images", n_results: int = 5,
                db_path: str | Path = "data/chroma", device: str = "cpu") -> list[dict]:
    model_id, revision = _model_for(db_path, target)
    vector = CLIPEmbedder(model_id=model_id, revision=revision, device=device).texts([query])[0]
    return _search(vector, target, db_path, n_results)


def search_image(image_path: str | Path, target: str = "captions", n_results: int = 5,
                 db_path: str | Path = "data/chroma", device: str = "cpu") -> list[dict]:
    model_id, revision = _model_for(db_path, target)
    vector = CLIPEmbedder(model_id=model_id, revision=revision, device=device).images([image_path])[0]
    return _search(vector, target, db_path, n_results)
