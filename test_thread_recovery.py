"""Tests for thread refresh after gateway timeout (524) during message send."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import requests

import thread_recovery as tr


class RecoverableSendFailureTests(unittest.TestCase):
    def test_http_524_is_recoverable(self):
        resp = MagicMock()
        resp.status_code = 524
        exc = requests.exceptions.HTTPError(response=resp)
        self.assertTrue(tr.is_recoverable_send_failure(exc))

    def test_http_400_is_not_recoverable(self):
        resp = MagicMock()
        resp.status_code = 400
        exc = requests.exceptions.HTTPError(response=resp)
        self.assertFalse(tr.is_recoverable_send_failure(exc))

    def test_timeout_is_recoverable(self):
        self.assertTrue(tr.is_recoverable_send_failure(requests.exceptions.Timeout()))


class ResolveThreadIdTests(unittest.TestCase):
    def test_prefers_partial_result_thread_id(self):
        threads = [{"thread_id": "newest"}, {"thread_id": "older"}]
        tid = tr.resolve_thread_id_after_send_failure(
            tr.NEW_THREAD,
            threads,
            {"thread_id": "from-sse"},
        )
        self.assertEqual(tid, "from-sse")

    def test_keeps_known_pending_thread_over_newest(self):
        threads = [{"thread_id": "newest"}, {"thread_id": "existing"}]
        tid = tr.resolve_thread_id_after_send_failure(
            "existing",
            threads,
            None,
        )
        self.assertEqual(tid, "existing")

    def test_new_thread_falls_back_to_newest_list_entry(self):
        threads = [{"thread_id": "newest"}]
        tid = tr.resolve_thread_id_after_send_failure(tr.NEW_THREAD, threads, None)
        self.assertEqual(tid, "newest")


class ConvertLgMessagesTests(unittest.TestCase):
    def test_maps_human_and_ai(self):
        out = tr.convert_lg_messages(
            [
                {"type": "human", "content": "hi", "id": "1"},
                {"type": "ai", "content": "hello", "id": "2"},
                {"type": "tool", "content": "ignored"},
            ]
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["role"], "user")
        self.assertEqual(out[1]["role"], "assistant")

    def test_carries_response_metadata_so_reports_survive_a_reload(self):
        """Created reports and plots ride on response_metadata; dropping it loses them."""
        artifacts = [
            {
                "name": "plot.png",
                "mime_type": "image/png",
                "encoding": "base64",
                "content": "aGVsbG8=",
                "size_bytes": 5,
            }
        ]
        out = tr.convert_lg_messages(
            [
                {
                    "type": "ai",
                    "content": "Your step count is trending down.",
                    "id": "2",
                    "response_metadata": {"created_artifacts": artifacts},
                }
            ]
        )
        self.assertEqual(out[0]["metadata"]["created_artifacts"], artifacts)

    def test_metadata_defaults_to_empty_dict(self):
        out = tr.convert_lg_messages([{"type": "ai", "content": "hi", "id": "1"}])
        self.assertEqual(out[0]["metadata"], {})


class SimulatedSendFlowTests(unittest.TestCase):
    """Simulate POST /message failing with 524 then pulling the thread."""

    def test_send_524_then_thread_pull_retrieves_assistant_message(self):
        thread_id = "b1c4668e-d232-4b0d-b7de-7d169fef48a4"
        pending = tr.NEW_THREAD
        thread_messages = {
            pending: [
                {
                    "role": "user",
                    "content": "architecture correction",
                    "response_time_ms": None,
                },
            ],
        }

        resp = MagicMock()
        resp.status_code = 524
        exc = requests.exceptions.HTTPError(response=resp)
        self.assertTrue(tr.is_recoverable_send_failure(exc))

        threads = [{"thread_id": thread_id}]
        resolved = tr.resolve_thread_id_after_send_failure(pending, threads, None)
        self.assertEqual(resolved, thread_id)

        backend_lg = [
            {"type": "human", "content": "architecture correction", "id": "h1"},
            {"type": "ai", "content": "Got it — learned.", "id": "a1"},
        ]
        ui_messages = tr.convert_lg_messages(backend_lg)
        tr.apply_recovered_messages(
            thread_messages,
            pending_thread_id=pending,
            final_tid=thread_id,
            messages=ui_messages,
        )

        self.assertNotIn(pending, thread_messages)
        assistant_msgs = [m for m in thread_messages[thread_id] if m["role"] == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0]["content"], "Got it — learned.")

    def test_langsmith_fixture_thread_id_and_reply_shape(self):
        """Regression: run from learned_information_error JSON metadata + output."""
        thread_id = "b1c4668e-d232-4b0d-b7de-7d169fef48a4"
        expected_reply = (
            "Got it — learned.\n\n"
            "Neural Nexus uses an **embedding model**, a **vectorstore**, and a "
            "**Postgres** database for retrieval/context. We **don’t** use **Chroma** "
            "and we **don’t** use **Llama**."
        )
        backend_lg = [
            {"type": "human", "content": "user correction", "id": "h1"},
            {"type": "ai", "content": expected_reply, "id": "a1"},
        ]
        thread_messages: dict = {tr.NEW_THREAD: [{"role": "user", "content": "user correction"}]}
        tr.apply_recovered_messages(
            thread_messages,
            pending_thread_id=tr.NEW_THREAD,
            final_tid=thread_id,
            messages=tr.convert_lg_messages(backend_lg),
        )
        self.assertIn("learned", thread_messages[thread_id][-1]["content"].lower())
        self.assertEqual(thread_messages[thread_id][-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
