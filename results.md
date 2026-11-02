# LensRAG measured retrieval results

Run UTC: 2026-11-02T20:28:07.739800+00:00

Corpus: 2,000 Flickr8k images; 9,900 indexed captions. Queries: 100.
Model: `openai/clip-vit-base-patch32` at `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; device `cpu`.

| k | Correct image retrieved | Recall@k |
|---|---:|---:|
| 1 | 40/100 | 0.400 |
| 5 | 64/100 | 0.640 |
| 10 | 76/100 | 0.760 |

## Methodology

First 2,000 distinct images in pinned shard `data/train-00000-of-00003.parquet` of `tsystems/flickr8k` at `81fc5f3a41274c80f17b0406426d57cac57ce6fb`; 100 query images sampled with Python random seed 42. Caption 0 is held out for each selected image. Query captions are excluded from the caption index. All correct images remain in the image index. CLIP is frozen, with normalized embeddings and cosine HNSW retrieval in persistent ChromaDB. No tuning is performed on these queries.

Recall@k is the fraction of queries whose single annotated paired image appears among the first k image results. Other plausible images receive no credit. This measures a specific fixed corpus, not arbitrary real-world search or unseen-data generalization. Flickr8k may overlap with CLIP pretraining; no claim of pretraining disjointness is made.

HNSW search is approximate. Small ranking changes were observed after reopening the database; the committed evidence uses the reopened index. See `verification.md` for the observed repeat-check results. Exact scores can also depend on library versions, hardware, and preprocessing.

## Qualitative failure review

12 of the first 12 top-1 failures in query order have visual review notes. These are qualitative observations, not a statistically representative taxonomy. Review was performed by the coding assistant inspecting the image contact sheet; it is not independent human annotation.

- Action and participant mismatch: 1
- Action mismatch: 2
- Ambiguous caption / alternate scene: 1
- Attribute and context mismatch: 1
- Context mismatch: 2
- Plausible alternate match: 3
- Spatial relationship mismatch: 2

| Query | Category | Observation |
|---|---|---|
| q-0003 | Action mismatch | Both images show a brown dog in snow. The paired dog is twisted mid-shake; the top result faces forward and does not visibly show that action. |
| q-0004 | Attribute and context mismatch | The paired man wears sunglasses and a light shirt. The top result emphasizes a sunset horizon on a boat but the man has no visible sunglasses and a darker shirt. |
| q-0005 | Action mismatch | The paired black dog is swimming with an object at its mouth. The top result shows a black dog airborne from a dock rather than swimming. |
| q-0006 | Spatial relationship mismatch | The paired child sits on the man's shoulders. The top result shows a child held overhead with raised arms, a similar silhouette but different support relationship. |
| q-0010 | Context mismatch | The top result has backpackers on a snowy wooded trail. The paired photo has a small hiker below a prominent snowy mountain; the forest result misses that mountain context. |
| q-0014 | Plausible alternate match | Both photos show multiple climbers on rock. The top result plausibly matches the broad caption; exact paired-image scoring still marks it wrong. Not a clear semantic error. |
| q-0015 | Context mismatch | The top result shows two children on a wooded boardwalk but no visible pond. The paired image includes a dark pond beside the children. |
| q-0018 | Action and participant mismatch | The paired photo shows two boys close to the camera, one with tongue out. The top result shows one boy by outdoor steps, without the tongue-out expression or second boy. |
| q-0019 | Spatial relationship mismatch | The top result contains one person on a bench. It misses the three-person arrangement and the behind relationship specified by the query. |
| q-0020 | Ambiguous caption / alternate scene | The query is grammatically unclear. Both scenes include seated people, but the top result shows adults near a stroller and does not preserve the paired foreground person in black shorts. |
| q-0021 | Plausible alternate match | The top result shows children moving on grass and plausibly satisfies the broad query. The paired image is a different group of running children; one-pair labels penalize this alternate. |
| q-0022 | Plausible alternate match | Both paired and top-result images show a dog on green sloping grass. The paired image has a steep mountainside; the caption itself is broad enough to describe the top result. |

## Reproducibility

Manifest SHA-256: `9b37b00fc2e56a6e82f2fd10ebcce4d60507baa47e23eb8d7d221a09a6e9d3c4`
Queries SHA-256: `6b9b7db79a5d55c6d4f43cabeaa577a61b79f89d5e77ae1d2af981422e5b893e`

Environment: `macOS-27.0-arm64-arm-64bit`

```json
{
  "torch": "2.15.0",
  "transformers": "4.57.6",
  "chromadb": "1.5.9",
  "numpy": "2.5.4",
  "pillow": "13.0.0"
}
```

Per-query ranks and retrieved IDs/distances are in `evaluation.json`; review evidence is in `failure_review.json` and `failure_contact_sheet.jpg`.
