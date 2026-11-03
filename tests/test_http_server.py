import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from memocore.embeddings import HashingEmbedder
from memocore.http_server import build_app
from memocore.retrieval.rerank import LexicalReranker
from memocore.store import MemoryStore


class HttpServerTest(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp(prefix="memocore-http-"))
        self.addCleanup(shutil.rmtree, root, True)
        self.store = MemoryStore(root / "memory.sqlite", HashingEmbedder(), LexicalReranker())
        self.addCleanup(self.store.close)

    def client(self, token=None):
        return TestClient(build_app(self.store, token=token))

    def test_remember_recall_read_forget(self):
        client = self.client()
        stored = client.post("/users/ann/memories", json={"messages": [
            {"text": "I adopted a beagle named Max yesterday", "at": "2023-05-08T13:56:00"},
            {"text": "My sister lives in Lisbon", "at": "2023-05-08T13:58:00"},
        ]})
        self.assertEqual(stored.status_code, 200)
        self.assertEqual([m["id"] for m in stored.json()], ["D1:1", "D1:2"])
        self.assertEqual(stored.json()[0]["time_note"], "time: yesterday = 7 May 2023")

        recall = client.get("/users/ann/recall", params={"query": "Which beagle did I adopt?", "budget": 1}).json()
        self.assertEqual([m["id"] for m in recall["memories"]], ["D1:1"])
        self.assertIn("[D1:1] user: I adopted a beagle", recall["context"])
        self.assertEqual(client.get("/users/bob/recall", params={"query": "dog"}).json()["memories"], [])  # isolated

        read = client.get("/users/ann/memories", params={"ids": "D1:2, D9:9"}).json()
        self.assertEqual([m["text"] for m in read], ["My sister lives in Lisbon"])
        self.assertEqual(client.get("/users/ann/memories", params={"ids": "nonsense"}).status_code, 422)

        self.assertEqual(client.delete("/users/ann/memories").status_code, 422)  # needs ids or everything=true
        self.assertEqual(client.delete("/users/ann/memories", params={"ids": "D1:1"}).json(), {"deleted": 1})
        self.assertEqual(client.delete("/users/ann/memories", params={"everything": "true"}).json(), {"deleted": 1})

    def test_validation_and_token(self):
        client = self.client()
        self.assertEqual(client.post("/users/ann/memories", json={"messages": []}).status_code, 422)
        self.assertEqual(client.get("/users/ann/recall", params={"query": "x", "budget": 99}).status_code, 422)
        guarded = self.client(token="s3cret")
        self.assertEqual(guarded.get("/health").status_code, 401)
        self.assertEqual(guarded.get("/health", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(guarded.get("/health", headers={"Authorization": "Bearer s3cret"}).json(), {"status": "ok"})
