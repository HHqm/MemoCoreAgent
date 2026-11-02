import unittest

import numpy as np

from memocore.embeddings import CachedEmbedder, HashingEmbedder
from tests.helpers import temp_settings


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.encoded: list[str] = []

    def encode(self, texts):
        self.encoded.extend(texts)
        return super().encode(texts)


class CachedEmbedderTest(unittest.TestCase):
    def test_only_new_texts_reach_the_model_and_vectors_match(self):
        settings = temp_settings(self)
        inner = CountingEmbedder()
        cached = CachedEmbedder(inner, settings.embedding_cache_path)
        first = cached.encode(["violin", "pottery"])
        second = cached.encode(["pottery", "violin", "painting"])
        self.assertEqual(inner.encoded, ["violin", "pottery", "painting"])
        self.assertEqual((cached.hits, cached.misses), (2, 3))
        np.testing.assert_allclose(second[1], first[0])
        np.testing.assert_allclose(second[2], HashingEmbedder().encode(["painting"])[0])

    def test_cache_survives_a_new_instance(self):
        settings = temp_settings(self)
        CachedEmbedder(HashingEmbedder(), settings.embedding_cache_path).encode(["violin"])
        inner = CountingEmbedder()
        CachedEmbedder(inner, settings.embedding_cache_path).encode(["violin"])
        self.assertEqual(inner.encoded, [])


if __name__ == "__main__":
    unittest.main()
