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
        return json.dumps(self.decisions.pop(0))


class AgenticRagTests(unittest.TestCase):
    def setUp(self):
        self.sripetch_chunks = [
            chunk("sripetch", "Sripetch_vs_SEC", 0, "Opening issue in the case."),
            chunk("sripetch", "Sripetch_vs_SEC", 1, "Petitioner's central argument."),
            chunk("sripetch", "Sripetch_vs_SEC", 2, "Government response."),
            chunk("sripetch", "Sripetch_vs_SEC", 3, "Questions from the Court."),
        ]
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

    def _decision(self, action, **overrides):
        payload = {
            "action": action,
            "query": "",
            "source_keys": [],
            "source_key": "",
            "offset": 0,
            "limit": 5,
            "chunk_id": -1,
            "radius": 2,
            "answer": "",
            "citation_ids": [],
            "clarification_question": "",
        }
        payload.update(overrides)
        return payload

    @patch("src.agentic_rag.search_corpus")
    def test_obvious_title_typo_can_trigger_broader_source_read(self, search):
        search.return_value = [self.sripetch_chunks[1], self.sripetch_chunks[2]]
        llm = ScriptedLLM([
            self._decision(
                "read_source",
                source_key="sripetch",
                offset=0,
                limit=4,
            ),
            self._decision(
                "answer",
                answer="The recording centers on the dispute described in the cited passages.",
                citation_ids=["Sripetch_vs_SEC:0", "Sripetch_vs_SEC:2"],
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
        self.assertEqual({item["source_key"] for item in result.evidence}, {"sripetch"})
        self.assertTrue(any(step.action == "read_source" for step in result.trace))

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

    @patch("src.agentic_rag.search_corpus")
    def test_unselected_source_tool_request_is_blocked(self, search):
        search.return_value = [self.sripetch_chunks[0]]
        llm = ScriptedLLM([
            self._decision(
                "search_transcripts",
                query="mars autonomy",
                source_keys=["nasa"],
                limit=5,
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
        # Only the bootstrap search should have executed. The attempted NASA
        # search is outside the selected boundary and must not hit retrieval.
        self.assertEqual(search.call_count, 1)
        self.assertIn("blocked", result.trace[1].detail.lower())

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


if __name__ == "__main__":
    unittest.main()
