"""Measured paired retrieval, inspectable failures, and evidence-based reports."""
from __future__ import annotations
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def recall_at_k(rankings, correct_ids, ks=(1, 5, 10)):
    if len(rankings) != len(correct_ids) or not correct_ids:
        raise ValueError('Need equally sized, nonempty rankings and ground truth')
    return {str(k): sum(correct in ranked[:k] for ranked, correct in zip(rankings, correct_ids)) / len(correct_ids) for k in ks}


def validate_index(images, captions, manifest, queries):
    """Fail closed on stale, partial, or incompatible collections before inference."""
    from .index import _expected
    records, expected_captions = _expected(manifest, queries)
    image_ids = {r['image_id'] for r in records}
    caption_ids = {r[0] for r in expected_captions}
    for collection, expected in ((images, image_ids), (captions, caption_ids)):
        if collection.count() != len(expected) or set(collection.get(include=[])['ids']) != expected:
            raise ValueError(f'{collection.name} is incomplete or stale; run lensrag index')
        meta = collection.metadata or {}
        if meta.get('manifest_digest') != manifest['digest']:
            raise ValueError(f'{collection.name} manifest metadata differs')
        if not meta.get('model_id') or not meta.get('model_revision'):
            raise ValueError(f'{collection.name} model metadata is missing')
    if any(images.metadata[k] != captions.metadata[k] for k in ('model_id', 'model_revision')):
        raise ValueError('Image and caption collection model metadata differs')


def evaluate(data_dir=Path('data'), db_path=Path('data/chroma'), output_dir=Path('artifacts'), device='cpu'):
    import chromadb
    from .clip import CLIPEmbedder
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    from .dataset import load_manifest, load_queries
    manifest = load_manifest(data_dir)
    queries = load_queries(data_dir, manifest)
    if not queries:
        raise ValueError('Evaluation requires at least one held-out query')
    client = chromadb.PersistentClient(path=str(db_path))
    images = client.get_collection('lensrag_images', embedding_function=None)
    captions = client.get_collection('lensrag_captions', embedding_function=None)
    validate_index(images, captions, manifest, queries)
    meta = images.metadata
    embedder = CLIPEmbedder(model_id=meta['model_id'], revision=meta['model_revision'], device=device)
    vectors = embedder.texts([q['text'] for q in queries])
    raw = images.query(query_embeddings=vectors.tolist(), n_results=min(10, images.count()), include=['metadatas', 'distances'])
    details = []
    for i, q in enumerate(queries):
        ranked = [dict(m, distance=float(d)) for m, d in zip(raw['metadatas'][i], raw['distances'][i])]
        ids = [r['image_id'] for r in ranked]
        details.append(dict(q, rank=(ids.index(q['image_id'])+1 if q['image_id'] in ids else None), retrieved=ranked))
    metrics = recall_at_k([[r['image_id'] for r in d['retrieved']] for d in details], [q['image_id'] for q in queries])
    evidence = {
        'run_utc': datetime.now(timezone.utc).isoformat(), 'dataset': manifest['dataset_repo'],
        'dataset_revision': manifest['dataset_revision'], 'source_shard': manifest['source_shard'],
        'seed': manifest['seed'], 'max_images': manifest['max_images'],
        'image_count': images.count(), 'caption_count': captions.count(), 'query_count': len(queries),
        'model_id': meta['model_id'], 'model_revision': meta['model_revision'], 'index_metadata': meta,
        'manifest_sha256': hashlib.sha256((data_dir/'manifest.json').read_bytes()).hexdigest(),
        'queries_sha256': hashlib.sha256((data_dir/'queries.json').read_bytes()).hexdigest(),
        'recall_at_k': metrics, 'device': device, 'platform': platform.platform(),
        'versions': {p: version(p) for p in ['torch', 'transformers', 'chromadb', 'numpy', 'pillow']},
        'queries': details,
    }

    # First 12 top-1 errors in the preselected query order; never cherry-picked by category.
    failures = [d for d in details if d['rank'] != 1][:12]
    template = [{'query_id': d['query_id'], 'correct_image_id': d['image_id'], 'top1_image_id': d['retrieved'][0]['image_id'], 'category': '', 'observation': ''} for d in failures]
    # Preserve existing reviews only if they refer to identical results.
    review_path = output_dir/'failure_review.json'
    if review_path.exists():
        old = json.loads(review_path.read_text())
        if [(r['query_id'], r['correct_image_id'], r['top1_image_id']) for r in old] != [(r['query_id'], r['correct_image_id'], r['top1_image_id']) for r in template]:
            raise ValueError('Existing review differs; move it aside before evaluating again')

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'evaluation.json').write_text(json.dumps(evidence, indent=2)+'\n')
    if not review_path.exists():
        review_path.write_text(json.dumps(template, indent=2)+'\n')
    records = {r['image_id']: r for r in manifest['records']}
    from PIL import Image, ImageDraw, ImageOps
    tiles = []
    for d in failures:
        tile = Image.new('RGB', (1000, 330), 'white')
        draw = ImageDraw.Draw(tile)
        import textwrap
        draw.text((8, 5), '\n'.join(textwrap.wrap(f"{d['query_id']}: {d['text']}", width=130)), fill='black')
        for j, (label, image_id) in enumerate([('Correct', d['image_id'])] + [(f'Top {i+1}', r['image_id']) for i, r in enumerate(d['retrieved'][:2])]):
            rec = records[image_id]
            path = data_dir / rec['image_path']
            with Image.open(path) as im:
                im = ImageOps.contain(im.convert('RGB'), (320, 245))
                tile.paste(im, (j*333+5, 65))
            draw.text((j*333+5, 45), f'{label}: {image_id}', fill='black')
        tiles.append(tile)
    if tiles:
        sheet = Image.new('RGB', (1000, 330*len(tiles)), 'white')
        for i, tile in enumerate(tiles): sheet.paste(tile, (0, i*330))
        sheet.save(output_dir/'failure_contact_sheet.jpg')
    generate_report(output_dir)
    return {k:v for k,v in evidence.items() if k != 'queries'}


def generate_report(output_dir=Path('artifacts')):
    output_dir = Path(output_dir)
    e = json.loads((output_dir/'evaluation.json').read_text())
    reviews = json.loads((output_dir/'failure_review.json').read_text()) if (output_dir/'failure_review.json').exists() else []
    valid = {d['query_id']:d for d in e['queries'] if d['rank'] != 1}
    seen = set()
    for r in reviews:
        if r['query_id'] in seen or r['query_id'] not in valid:
            raise ValueError('Review must contain unique top-1 failures from this run')
        seen.add(r['query_id'])
        d = valid[r['query_id']]
        if r['correct_image_id'] != d['image_id'] or r['top1_image_id'] != d['retrieved'][0]['image_id']:
            raise ValueError('Review does not match actual retrieval')
    complete = [r for r in reviews if r.get('category') and r.get('observation')]
    lines = ['# LensRAG measured retrieval results', '', f"Run UTC: {e['run_utc']}", '',
        f"Corpus: {e['image_count']:,} Flickr8k images; {e['caption_count']:,} indexed captions. Queries: {e['query_count']}.",
        f"Model: `{e['model_id']}` at `{e['model_revision']}`; device `{e['device']}`.", '',
        '| k | Correct image retrieved | Recall@k |', '|---|---:|---:|']
    for k, score in e['recall_at_k'].items():
        count = sum(d['rank'] is not None and d['rank'] <= int(k) for d in e['queries'])
        lines.append(f'| {k} | {count}/{e["query_count"]} | {score:.3f} |')
    lines += ['', '## Methodology', '',
        f"First {e['image_count']:,} distinct images in pinned shard `{e['source_shard']}` of `{e['dataset']}` at `{e['dataset_revision']}`; {e['query_count']} query images sampled with Python random seed {e['seed']}. Caption 0 is held out for each selected image. Query captions are excluded from the caption index. All correct images remain in the image index. CLIP is frozen, with normalized embeddings and cosine HNSW retrieval in persistent ChromaDB. No tuning is performed on these queries.", '',
        'Recall@k is the fraction of queries whose single annotated paired image appears among the first k image results. Other plausible images receive no credit. This measures a specific fixed corpus, not arbitrary real-world search or unseen-data generalization. Flickr8k may overlap with CLIP pretraining; no claim of pretraining disjointness is made.', '',
        'HNSW search is approximate. Small ranking changes were observed after reopening the database; the committed evidence uses the reopened index. See `verification.md` for the observed repeat-check results. Exact scores can also depend on library versions, hardware, and preprocessing.', '',
        '## Qualitative failure review', '',
        f'{len(complete)} of the first {len(reviews)} top-1 failures in query order have visual review notes. These are qualitative observations, not a statistically representative taxonomy. Review was performed by the coding assistant inspecting the image contact sheet; it is not independent human annotation.', '']
    if complete:
        for cat, count in sorted(Counter(r['category'] for r in complete).items()): lines.append(f'- {cat}: {count}')
        lines += ['', '| Query | Category | Observation |', '|---|---|---|']
        for r in complete: lines.append('| '+' | '.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ['query_id','category','observation'])+' |')
    else: lines.append('Visual review pending. No failure categories have been asserted.')
    lines += ['', '## Reproducibility', '', f"Manifest SHA-256: `{e['manifest_sha256']}`", f"Queries SHA-256: `{e['queries_sha256']}`", '', f"Environment: `{e['platform']}`", '', '```json', json.dumps(e['versions'], indent=2), '```', '', 'Per-query ranks and retrieved IDs/distances are in `evaluation.json`; review evidence is in `failure_review.json` and `failure_contact_sheet.jpg`.']
    (output_dir/'results.md').write_text('\n'.join(lines)+'\n')
    return str(output_dir/'results.md')
