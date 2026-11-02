# LensRAG

A working CLIP-based multimodal retrieval project: images and captions share an embedding space, stored in two persistent ChromaDB collections. Search with text to retrieve images, or with an image to retrieve captions (or similar images). Despite the project name, this is retrieval only; it does not generate answers with an LLM.

## Dataset and model

- **Flickr8k: exactly 2,000 images and 10,000 original captions.** The first 2,000 distinct rows of shard `data/train-00000-of-00003.parquet` from [tsystems/flickr8k](https://huggingface.co/datasets/tsystems/flickr8k), pinned at `81fc5f3a41274c80f17b0406426d57cac57ce6fb`. This is a convenience subset, not the full dataset or an official benchmark split.
- **OpenAI CLIP ViT-B/32**, [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32), revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Frozen image and text encoders, 512-dimensional unit-normalized vectors. No training or fine-tuning.
- **ChromaDB on disk**, cosine HNSW search, `lensrag_images` and `lensrag_captions`. Embeddings are supplied explicitly; Chroma does not choose a different default embedding model. Source filenames and captions accompany retrieved IDs.

The public mirror labels its dataset MIT; that label does not establish ownership of every underlying Flickr photo. Images are downloaded locally and are not bundled in the repository. Respect the original dataset/image terms and provenance when redistributing or publishing examples. The project code is MIT licensed.

## Run locally

Use Python 3.12 (supported range 3.11–3.13), with `uv` installed. Run commands from the repository root. CPU is the default; no GPU or paid API is required. Initial setup downloads the image shard and CLIP checkpoint; later runs can use the local caches. Allow several GB of disk space and time for CPU embedding.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-lock.txt
uv pip install -e .
export HF_HOME="$PWD/.cache/huggingface"
export ANONYMIZED_TELEMETRY=False
lensrag prepare
lensrag index
lensrag stats
lensrag text "a dog jumping through the snow" -k 5
lensrag image data/images/img-000000.jpg -k 5
lensrag image data/images/img-000000.jpg --target images -k 5
lensrag evaluate --output-dir artifacts/reproduced
```

Search results are JSON with image IDs, image paths relative to `data/`, captions (when searching captions), and cosine distances (smaller is closer). Image-to-image queries can return the input itself if it belongs to the corpus. A different local JPEG/PNG can also be used as input. Image-to-text retrieval returns existing indexed captions; it does not write a new caption.

`index` resumes by skipping existing vector IDs and rejects mismatching index provenance. `prepare --rebuild` replaces the prepared manifest; `index --rebuild` explicitly rebuilds the two project collections. These options are not needed for the standard run. For an already downloaded pinned shard, use `lensrag prepare --parquet data/flickr8k-shard.parquet`.

For a different subset, use `prepare --max-images N --query-count Q --rebuild`, followed by `index --rebuild`; its scores must be reported as a separate experiment. The first shard limits the available subset size. Optional `--device mps` or `--device cuda` requires compatible hardware/PyTorch; the recorded run uses CPU.

## Evaluation

The 100 query images are selected using `random.Random(42).sample` from the fixed corpus. Caption 0 from each becomes a query with its paired image as the known correct match. These 100 captions are excluded from the caption index, leaving **9,900 indexed captions**. The **2,000 images remain in the image index**, because retrieving their paired images is the task. The query split is held out from caption indexing, not from pretrained CLIP's unknown training corpus.

Text-to-image recall@k is `queries with paired image in top k / 100`, for k = 1, 5, 10. No parameters or prompts are tuned on these queries. Alternate images that also satisfy the caption do not count as correct, so some measured misses are ambiguous rather than objectively irrelevant. Image-to-text search is demonstrated; no image-to-text recall claim is made.

Measured CPU run (100 queries, 2,000 candidate images):

| Metric | Correct queries | Recall |
|---|---:|---:|
| Recall@1 | 40/100 | 40% |
| Recall@5 | 64/100 | 64% |
| Recall@10 | 76/100 | 76% |

These are scores from reopening the persisted database. The initial run returned 41% recall@1; after reopening, both evaluation and the independent agent check returned 40%. Recall@5 and recall@10 were unchanged. HNSW is approximate, and the cause of the small ranking change was not isolated; these are observed run results, not a promise of bit-identical results on every platform.

See [the generated results](results.md) and [full per-query evidence](artifacts/evaluation.json). Visual inspection of the first 12 top-1 misses found action confusion (a swimming dog versus a dog jumping into water), lost attributes/context (sunglasses versus a sunset horizon), and spatial relationship errors (sitting on shoulders versus being held overhead). Several misses were plausible alternate images for broad captions such as “A dog on a green hillside.” The [review notes](artifacts/failure_review.json) distinguish those ambiguous matches from clear semantic errors. This sample is descriptive, not a measured population-wide failure distribution.

**These results apply only to this fixed Flickr8k subset and query set. They do not establish performance on arbitrary real-world image search, larger galleries, other domains, or new languages.** Flickr8k may overlap with CLIP's pretraining data. This is a portfolio retrieval experiment, not a new model or a general benchmark result.

## Inspect and reproduce the evidence

`artifacts/evaluation.json` stores every query, known correct ID, top-10 results, distances, and rank. `artifacts/results.md` is generated from that JSON and `failure_review.json`. The contact sheet shows the paired image and top retrieved images for the first 12 top-1 failures in the preselected query order. Categories are qualitative and may overlap.

To perform your own review, run evaluation into a separate output directory, inspect its contact sheet, fill `category` and `observation` in its `failure_review.json`, then run:

```bash
lensrag report --output-dir artifacts/reproduced
pytest -q
```

The root `results.md` is a copy of the generated `artifacts/results.md`; after updating the recorded review, regenerate with `lensrag report` and `cp artifacts/results.md results.md`. New reproduction runs should use a separate output directory to preserve this run. The manifest includes a local source path, so manifest hashes are run-specific even when query/image selection matches.

The recorded qualitative review is by the coding assistant inspecting actual images; it is not independent human annotation. Review those examples yourself before claiming in an interview that you personally reviewed them.

## Code map

- `dataset.py`: pinned shard download, extraction, fixed subset, held-out query selection.
- `clip.py`: pretrained image/text encoders and normalization.
- `index.py`: persistent image/caption collections, resumable indexing, both search directions.
- `evaluation.py`: recall, per-query evidence, contact sheet and report generation.
- `cli.py`: runnable demonstration interface.

Tests exercise real temporary Chroma persistence with small supplied vectors and verify held-out caption handling and recall arithmetic. They do not substitute for the separate real model/data evaluation.
