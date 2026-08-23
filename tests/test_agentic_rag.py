import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agentic_rag import REFUSAL_TEXT, agentic_ask


def chunk(source_key, source_id, chunk_id, text, start=0.0, end=10.0):
    return {
        "source_key": source_key,
        "source_id": source_id,
        "chunk_id": chunk_id,
        "text": text,
        "speaker_text": text,
        "start": start,
        "end": end,
    }


class ScriptedLLM:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if not self.decisions:
            raise AssertionError("LLM was called more times than expected")
        decision = self.decisions.pop(0)
        if isinstance(decision, str):
            return decision
        return json.dumps(decision)


class AgenticRagTests(unittest.TestCase):
    def setUp(self):
        self.sripetch_chunks = [
            chunk("sripetch", "Sripetch_vs_SEC", i, f"Sripetch case passage {i}.")
            for i in range(24)
        ]
        self.sripetch_chunks[0]["speaker_text"] = "Opening issue in the case."
        self.sripetch_chunks[7]["speaker_text"] = "Petitioner's central argument."
        self.sripetch_chunks[15]["speaker_text"] = "Government response."
        self.sripetch_chunks[23]["speaker_text"] = "Questions from the Court."

        self.nasa_chunks = [
            chunk("nasa", "NASA_Clip", 0, "The crew discusses autonomy during loss of signal."),
            chunk("nasa", "NASA_Clip", 1, "Mission Control is unavailable for two weeks."),
        ]
        self.index = SimpleNamespace(
            sources={
                "sripetch": SimpleNamespace(chunks=self.sripetch_chunks),
                "nasa": SimpleNamespace(chunks=self.nasa_chunks),
            }
        )
        self.catalog = [
            {"source_key": "sripetch", "display_name": "Sripetch vs SEC"},
            {"source_key": "nasa", "display_name": "NASA CHAPEA 2 - Clip 3"},
        ]

    def _request(self, tool, **overrides):
        payload = {"tool": tool}
        payload.update(overrides)
        return payload

    def _decision(self, action, **overrides):
        payload = {"action": action}
        payload.update(overrides)
        return payload

    @patch("src.agentic_rag.search_corpus")
    def test_title_typo_is_presampled_and_can_answer_in_one_llm_turn(self, search):
        search.return_value = [self.sripetch_chunks[7], self.sripetch_chunks[15]]
        llm = ScriptedLLM([
            self._decision(
                "answer",
                answer="The recording centers on the dispute described in the cited passages.",
                citation_ids=["Sripetch_vs_SEC:0", "Sripetch_vs_SEC:23"],
            ),
        ])

        result = agentic_ask(
            query="explain sreiptech vs sec",
            index=self.index,
            selected_source_keys=["sripetch", "nasa"],
            source_catalog=self.catalog,
            llm_client=llm,
        )

        self.assertTrue(result.answerable)
        self.assertEqual(result.outcome, "answer")
        self.assertEqual(llm.calls, 1)
        self.assertTrue(any(step.action == "metadata_sample" for step in result.trace))

    @patch("src.agentic_rag.search_corpus")
    def test_out_of_corpus_question_refuses_without_chatbot_reinterpretation(self, search):
        search.return_value = [self.sripetch_chunks[0]]
        llm = ScriptedLLM([
            self._decision("refuse"),
        ])

        result = agentic_ask(
            query="wdyk about mars and autonomy",
            index=self.index,
            selected_source_keys=["sripetch"],
            source_catalog=self.catalog,
            llm_client=llm,
        )

        self.assertFalse(result.answerable)
        self.assertEqual(result.outcome, "refusal")
        self.assertEqual(result.answer, REFUSAL_TEXT)
        self.assertEqual(result.citations, [])
        self.assertEqual(llm.calls, 1)

    @patch("src.agentic_rag.search_corpus")
    def test_clear_natural_language_can_answer_from_selected_nasa_evidence(self, search):
        search.return_value = self.nasa_chunks
        llm = ScriptedLLM([
            self._decision(
                "answer",
                answer="They act more cautiously because Mission Control is unavailable.",
                citation_ids=["NASA_Clip:0", "NASA_Clip:1"],
            ),
        ])

        result = agentic_ask(
            query="why do they act more cautiously when they cant talk to mission control",
            index=self.index,
            selected_source_keys=["nasa"],
            source_catalog=self.catalog,
            llm_client=llm,
        )

        self.assertTrue(result.answerable)
        self.assertEqual(len(result.citations), 2)
        self.assertEqual({item["source_key"] for item in result.evidence}, {"nasa"})
        self.assertEqual(llm.calls, 1)

    @patch("src.agentic_rag.search_corpus")
    def test_unselected_source_batch_request_is_blocked(self, search):
        search.return_value = [self.sripetch_chunks[0]]
        llm = ScriptedLLM([
            self._decision(
                "gather",
                requests=[
                    self._request(
                        "search_transcripts",
                        query="mars autonomy",
                        source_keys=["nasa"],
                        limit=5,
                    )
                ],
            ),
            self._decision("refuse"),
        ])

        result = agentic_ask(
            query="what does mars autonomy mean",
            index=self.index,
            selected_source_keys=["sripetch"],
            source_catalog=self.catalog,
            llm_client=llm,
        )

        self.assertEqual(result.outcome, "refusal")
        self.assertEqual(search.call_count, 1)
        self.assertTrue(any("blocked" in step.detail.lower() for step in result.trace))

    @patch("src.agentic_rag.search_corpus")
    def test_invalid_citation_downgrades_to_refusal(self, search):
        search.return_value = [self.sripetch_chunks[0]]
        llm = ScriptedLLM([
            self._decision(
                "answer",
                answer="Unsupported answer.",
                citation_ids=["Sripetch_vs_SEC:999"],
            ),
        ])

        result = agentic_ask(
            query="explain the case",
            index=self.index,
            selected_source_keys=["sripetch"],
            source_catalog=self.catalog,
            llm_client=llm,
        )

        self.assertFalse(result.answerable)
        self.assertEqual(result.answer, REFUSAL_TEXT)

    @patch("src.agentic_rag.search_corpus")
    def test_clarification_is_available_for_true_remaining_ambiguity(self, search):
        search.return_value = [self.sripetch_chunks[0]]
        llm = ScriptedLLM([
            self._decision(
                "clarify",
                clarification_question="Which speaker do you mean by John?",
            ),
        ])

        result = agentic_ask(
            query="what did John say about it",
            index=self.index,
            selected_source_keys=["sripetch"],
            source_catalog=self.catalog,
            llm_client=llm,
        )

        self.assertEqual(result.outcome, "clarification")
        self.assertIn("Which speaker", result.answer)

    @patch("src.agentic_rag.search_corpus")
    def test_multiple_retrieval_requests_share_one_controller_round(self, search):
        search.side_effect = [
            [self.sripetch_chunks[0]],
            [self.sripetch_chunks[7]],
            [self.sripetch_chunks[15]],
        ]
        llm = ScriptedLLM([
            self._decision(
                "gather",
                requests=[
                    self._request(
                        "search_transcripts",
                        query="petitioner position",
                        source_keys=["sripetch"],
                        limit=3,
                    ),
                    self._request(
                        "search_transcripts",
                        query="government position",
                        source_keys=["sripetch"],
                        limit=3,
                    ),
                ],
            ),
            self._decision(
                "answer",
                answer="The sides take different positions in the cited passages.",
                citation_ids=["Sripetch_vs_SEC:7", "Sripetch_vs_SEC:15"],
            ),
        ])

        result = agentic_ask(
            query="how do the two sides disagree",
            index=self.index,
            selected_source_keys=["sripetch"],
            source_catalog=self.catalog,
            llm_client=llm,
        )

        self.assertTrue(result.answerable)
        self.assertEqual(search.call_count, 3)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(
            len([step for step in result.trace if step.action == "search_transcripts"]),
            2,
        )

    @patch("src.agentic_rag.search_corpus")
    def test_controller_failure_is_not_mislabeled_as_missing_evidence(self, search):
        search.return_value = [self.sripetch_chunks[0]]
        llm = ScriptedLLM(["not-json"])

        result = agentic_ask(
            query="explain the case",
            index=self.index,
            selected_source_keys=["sripetch"],
            source_catalog=self.catalog,
            llm_client=llm,
        )

        self.assertEqual(result.outcome, "error")
        self.assertNotEqual(result.answer, REFUSAL_TEXT)
        self.assertTrue(any(step.action == "error" for step in result.trace))


if __name__ == "__main__":
    unittest.main()
