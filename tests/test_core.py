import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from lensrag.dataset import prepare, load_manifest
from lensrag.index import VectorIndex, build_index


def _jpeg(color):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="JPEG")
    return output.getvalue()


def test_prepare_and_persistent_index_holdout(tmp_path, monkeypatch):
    rows = [
        {"image": {"bytes": _jpeg("red"), "path": None},
         "image_filename": "red.jpg", "captions": ["red first", "red second"]},
        {"image": {"bytes": _jpeg("red"), "path": None},
         "image_filename": "red.jpg", "captions": ["duplicate"]},
        {"image": {"bytes": _jpeg("blue"), "path": None},
         "image_filename": "blue.jpg", "captions": ["blue first", "blue second"]},
    ]
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), shard)
    data_dir = tmp_path / "data"
    manifest = prepare(shard, data_dir, max_images=2, query_count=1)
    assert manifest["image_count"] == 2
    assert manifest["indexed_caption_count"] == 3
    assert [r["image_filename"] for r in manifest["records"]] == ["red.jpg", "blue.jpg"]
    assert load_manifest(data_dir)["digest"] == manifest["digest"]
    queries = json.loads((data_dir / "queries.json").read_text())

    class FakeEmbedder:
        def __init__(self, **kwargs):
            pass

        def texts(self, texts):
            return np.array([[1.0, 0.0] if "red" in text else [0.0, 1.0]
                             for text in texts], dtype=np.float32)

        def images(self, paths):
            return np.array([[1.0, 0.0] if "000000" in str(path) else [0.0, 1.0]
                             for path in paths], dtype=np.float32)

    monkeypatch.setattr("lensrag.index.CLIPEmbedder", FakeEmbedder)
    db_path = tmp_path / "chroma"
    stats = build_index(data_dir, db_path, batch_size=1)
    assert stats["images"] == 2
    assert stats["captions"] == 3
    reopened = VectorIndex(db_path)
    assert reopened.images.count() == 2
    assert reopened.captions.count() == 3
    heldout = queries[0]
    assert heldout["text"] not in reopened.captions.get(include=["documents"])["documents"]
    assert heldout["image_id"] in reopened.images.get(include=[])["ids"]
    hits = reopened.query(reopened.images, np.array([1.0, 0.0], dtype=np.float32), 1)
    assert hits[0]["image_id"] == "img-000000"
    assert hits[0]["distance"] < 1e-5
    assert build_index(data_dir, db_path) == stats


def test_tampered_queries_rejected_before_index_creation(tmp_path, monkeypatch):
    import hashlib
    import pytest
    from lensrag.dataset import _json_bytes, load_queries
    records = [{'image_id': 'a', 'captions': ['first', 'second']}]
    manifest = {'records': records, 'seed': 42, 'query_count': 1}
    manifest['digest'] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    original = [{'query_id': 'q-0000', 'image_id': 'a', 'text': 'first'}]
    (tmp_path / 'queries.json').write_text(json.dumps(original))
    assert load_queries(tmp_path, load_manifest(tmp_path)) == original
    original[0]['text'] = 'second'
    (tmp_path / 'queries.json').write_text(json.dumps(original))
    def forbidden(*args, **kwargs):
        raise AssertionError('Index must not be opened for invalid queries')
    monkeypatch.setattr('lensrag.index.VectorIndex', forbidden)
    with pytest.raises(ValueError, match='deterministic'):
        build_index(tmp_path, tmp_path / 'unused')
    manifest['seed'] = 43
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='digest mismatch'):
        load_manifest(tmp_path)
