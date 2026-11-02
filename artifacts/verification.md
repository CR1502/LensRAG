# Verification record

A real CPU run embedded 2,000 Flickr8k images and 9,900 non-held-out captions using the pinned CLIP checkpoint into persistent cosine Chroma collections.

The initial evaluation returned recall@1 = 0.41, recall@5 = 0.64, recall@10 = 0.76. An independent GPT-6 Astra check reopened Chroma, recomputed all 100 query embeddings, and returned 0.40 / 0.64 / 0.76. Six top-10 rankings changed; query q-0085 changed from paired image at rank 1 to rank 2. The cause was not isolated. HNSW is an approximate index; these observations should not be presented as bit-identical retrieval guarantees.

The final CLI evaluation was rerun against the reopened persisted database. **The committed `evaluation.json` and headline results use this final run: 0.40 / 0.64 / 0.76.** The first 12 top-1 failures were unchanged, so the image review still applies.

Independent checks also confirmed actual collection sizes, cosine distance configuration, manifest/query hashes, no query string appearing verbatim in indexed captions, and no duplicate extracted JPEG hashes. All nine offline tests passed. CLI text-to-image and image-to-caption smoke searches ran using the real frozen CLIP model and persisted collections.

The failure contact sheet was visually inspected by the primary coding assistant. Categories in `failure_review.json` are assistant-authored qualitative observations, not independent human annotations. Three of the reviewed failures are plausible alternate matches rather than clear semantic mistakes.

Agent roles used the available model names: GPT-6 Luna for scaffolding, GPT-6 Sol for implementation, GPT-6 Astra for independent verification, and GPT-6.1 Sol for corrections. The requested 5.6 model names were unavailable.
