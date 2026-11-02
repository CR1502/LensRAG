import pytest
from lensrag.evaluation import recall_at_k


def test_recall_uses_query_denominator_and_rank_cutoff():
    ranks = [['wrong', 'a'], ['b'], ['wrong']]
    assert recall_at_k(ranks, ['a', 'b', 'c'], ks=(1, 2, 10)) == {'1': 1/3, '2': 2/3, '10': 2/3}


def test_recall_rejects_empty_or_unmatched_ground_truth():
    with pytest.raises(ValueError):
        recall_at_k([], [])
    with pytest.raises(ValueError):
        recall_at_k([['a']], ['a', 'b'])


@pytest.mark.parametrize('fault', ['partial', 'stale', 'manifest', 'model'])
def test_index_guard_rejects_incomplete_or_incompatible_collections(fault):
    from lensrag.evaluation import validate_index
    manifest = {'digest': 'digest', 'records': [
        {'image_id': 'a', 'captions': ['heldout', 'indexed']} ]}
    queries = [{'image_id': 'a'}]

    class Collection:
        def __init__(self, name, ids):
            self.name, self.ids = name, ids
            self.metadata = {'manifest_digest': 'digest', 'model_id': 'model', 'model_revision': 'rev'}
        def count(self):
            return len(self.ids)
        def get(self, include):
            return {'ids': self.ids}

    images = Collection('images', ['a'])
    captions = Collection('captions', ['a::caption-1'])
    validate_index(images, captions, manifest, queries)
    if fault == 'partial':
        captions.ids = []
    elif fault == 'stale':
        images.ids = ['stale']
    elif fault == 'manifest':
        captions.metadata['manifest_digest'] = 'old'
    else:
        captions.metadata['model_revision'] = 'other'
    with pytest.raises(ValueError):
        validate_index(images, captions, manifest, queries)


def test_evaluate_partial_index_does_not_embed_or_write_outputs(tmp_path, monkeypatch):
    import hashlib
    import json
    from lensrag.dataset import _json_bytes
    from lensrag.evaluation import evaluate
    manifest = {'seed': 7, 'query_count': 1, 'records': [
        {'image_id': 'a', 'captions': ['heldout', 'indexed']} ]}
    manifest['digest'] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'queries.json').write_text(json.dumps([
        {'query_id': 'q-0000', 'image_id': 'a', 'text': 'heldout'}]))

    class Collection:
        name = 'partial'
        metadata = {'manifest_digest': manifest['digest'], 'model_id': 'model', 'model_revision': 'rev'}
        def count(self):
            return 0
    class Client:
        def get_collection(self, *args, **kwargs):
            return Collection()
    def forbidden(**kwargs):
        raise AssertionError('Must reject incomplete index before model loading')
    monkeypatch.setattr('chromadb.PersistentClient', lambda **kwargs: Client())
    monkeypatch.setattr('lensrag.clip.CLIPEmbedder', forbidden)
    output = tmp_path / 'outputs'
    with pytest.raises(ValueError, match='incomplete or stale'):
        evaluate(tmp_path, tmp_path / 'db', output)
    assert not output.exists()
