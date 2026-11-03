import asyncio
import shutil
from dataclasses import replace
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from memocore.datasets.locomo import load_locomo
from memocore.embeddings import HashingEmbedder
from memocore.mcp_server import build_server
from memocore.retrieval.rerank import LexicalReranker
from memocore.store import DEFAULT_OPTIONS, MemoryStore, format_session_time, parse_session_time
from memocore.systems import HybridRAGSystem
from tests.helpers import FIXTURE

MAY_8 = datetime(2023, 5, 8, 13, 56)
# Every read-time option on, so the store is checked against the evaluated system with all of them.
CONTEXTUAL = replace(DEFAULT_OPTIONS, context_before=1, time_candidates=20, speaker_boost=0.1)


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.encoded = 0

    def encode(self, texts):
        self.encoded += len(texts)
        return super().encode(texts)


class MemoryStoreTest(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp(prefix="memocore-store-"))
        self.addCleanup(shutil.rmtree, root, True)
        self.path = root / "memory.sqlite"
        self.store = self._open()

    def _open(self, embedder=None, options=DEFAULT_OPTIONS):
        store = MemoryStore(self.path, embedder or HashingEmbedder(), LexicalReranker(), options=options)
        self.addCleanup(store.close)
        return store

    def test_store_and_benchmark_runs_share_one_default_read_path(self):
        from memocore.runner import RunConfig

        from memocore.retrieval_eval import ABLATION_CONFIGS

        self.assertEqual(RunConfig(system="hybrid_rag", run_name="x").hybrid, DEFAULT_OPTIONS)
        self.assertEqual(list(ABLATION_CONFIGS.values())[-1], DEFAULT_OPTIONS)  # the ablation ends on the default

    def test_session_time_round_trips_the_locomo_format(self):
        self.assertEqual(format_session_time(MAY_8), "1:56 pm on 8 May, 2023")
        self.assertEqual(parse_session_time("12:05 am on 1 January, 2024"), datetime(2024, 1, 1, 0, 5))

    def test_ids_sessions_and_time_notes(self):
        first = self.store.add("u", "I adopted a dog yesterday", at=MAY_8)
        self.assertEqual(first.id, "D1:1")
        self.assertEqual(first.time_note, "time: yesterday = 7 May 2023")
        self.assertEqual(self.store.add("u", "His name is Max", at=MAY_8 + timedelta(minutes=10)).id, "D1:2")
        self.assertEqual(self.store.add("u", "Back again", at=MAY_8 + timedelta(hours=2)).id, "D2:1")
        self.assertEqual(self.store.add("u", "Packing", at=MAY_8 + timedelta(hours=3), session="trip").id, "D3:1")
        self.assertEqual(self.store.add("u", "Landed", at=MAY_8 + timedelta(days=1), session="trip").id, "D3:2")
        with self.assertRaises(ValueError):
            self.store.add("u", "   ")

    def test_search_picks_what_the_evaluated_system_picks(self):
        for options in (DEFAULT_OPTIONS, CONTEXTUAL):
            with self.subTest(contextual=options is CONTEXTUAL):
                self.store.close()
                self.path.unlink()
                self.store = self._open(options=options)
                self._check_against_reference(options)

    def _check_against_reference(self, options):
        embedder, reranker = HashingEmbedder(), LexicalReranker()
        for sample in load_locomo(FIXTURE):
            conversation = sample.conversation
            for session in conversation.sessions:
                for turn in session.turns:
                    memory = self.store.add(
                        conversation.sample_id, turn.text, speaker=turn.speaker, image=turn.image_caption,
                        at=parse_session_time(session.date_time), session=str(session.index),
                    )
                    self.assertEqual(memory.id, turn.dia_id)
            reference = HybridRAGSystem(embedder, reranker, options=options)
            reference.ingest(conversation)
            for qa in sample.qa:
                expected = reference.turn_ids(reference.select(reference.rank([qa.question])[0], 3))
                recall = self.store.search(conversation.sample_id, qa.question, budget=3)
                self.assertEqual({m.id for m in recall.memories}, set(expected), qa.question)
                self.assertIn(f"[{recall.memories[0].id}] ", recall.context)
        self.assertEqual(self.store.search("someone-else", "anything").memories, ())

    def test_reopening_reuses_stored_vectors(self):
        self.store.add("u", "I play the violin", at=MAY_8)
        self.store.add("u", "I live in Denver", at=MAY_8)
        self.store.close()
        counting = CountingEmbedder()
        reopened = self._open(counting)
        self.assertEqual([m.id for m in reopened.search("u", "Where do I live?", budget=1).memories], ["D1:2"])
        self.assertEqual(counting.encoded, 1)  # only the query

    def test_contextual_vectors_are_stored_and_refreshed_after_a_forget(self):
        self.store.close()
        store = self._open(options=CONTEXTUAL)
        for text in ("Do you play anything?", "Yes, the violin", "Since last spring"):
            store.add("u", text, at=MAY_8)
        store.close()
        counting = CountingEmbedder()
        reopened = self._open(counting, CONTEXTUAL)
        reopened.search("u", "Which instrument?")
        self.assertEqual(counting.encoded, 1)  # only the query: stored vectors already carry their context
        reopened.forget("u", ["D1:2"])
        counting.encoded = 0
        reopened.search("u", "Which instrument?")
        self.assertEqual(counting.encoded, 2)  # the query, and D1:3 whose context changed
        with self.assertRaises(ValueError):
            MemoryStore(self.path, HashingEmbedder(), None, options=replace(DEFAULT_OPTIONS, context_after=1))

    def test_forget_deletes_the_text_from_the_file(self):
        self.store.add("u", "my locker code is zebra-pancake-4711", at=MAY_8)
        self.store.add("u", "I like tea", at=MAY_8)
        self.assertEqual(self.store.forget("u", ["D1:1", "D9:9"]), 1)
        self.assertEqual(self.store.get("u", ["D1:1"]), [])
        self.assertNotIn("D1:1", {m.id for m in self.store.search("u", "locker code").memories})
        self.assertNotIn(b"zebra-pancake-4711", self.path.read_bytes())
        # Ids are never reused, so an old citation cannot point at a different message.
        self.assertEqual(self.store.add("u", "and coffee", at=MAY_8).id, "D1:3")
        self.assertEqual(self.store.forget("u"), 2)
        self.assertEqual(self.store.count("u"), 0)
        self.assertNotIn(b"I like tea", self.path.read_bytes())
        with self.assertRaises(ValueError):
            self.store.forget("u", ["not-an-id"])

    def test_mcp_tools(self):
        server = build_server(self.store, "me")

        def call(name, **arguments):
            result = asyncio.run(server.call_tool(name, arguments))
            content = getattr(result, "content", None)  # mcp 2.x returns a CallToolResult
            if content is None:  # mcp 1.x: the content list, or (content, structured output)
                content = result[0] if isinstance(result, tuple) else result
            return content[0].text

        names = {tool.name for tool in asyncio.run(server.list_tools())}
        self.assertEqual(names, {"remember", "recall", "read_memories", "forget"})
        self.assertIn("[D1:1]", call("remember", text="I started learning the violin last week"))
        self.assertIn("I started learning the violin", call("recall", query="What instrument am I learning?"))
        self.assertIn("Not found: D7:7", call("read_memories", ids=["D1:1", "D7:7"]))
        self.assertEqual(call("forget", ids=["D1:1"]), "Deleted 1 of 1 messages.")
        self.assertEqual(call("recall", query="violin"), "No memories stored yet.")
