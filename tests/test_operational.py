"""Tests for transcript validation and the local decision audit log."""
import json
import tempfile
import unittest
from pathlib import Path

from operational import append_decision_event, is_timeout_error, is_usable_transcript


class OperationalTests(unittest.TestCase):
    def test_transcript_requires_two_tokens(self):
        self.assertFalse(is_usable_transcript(""))
        self.assertFalse(is_usable_transcript("help"))
        self.assertTrue(is_usable_transcript("please help"))

    def test_wrapped_timeout_is_detected(self):
        self.assertTrue(is_timeout_error(RuntimeError("recognition request timed out")))
        self.assertFalse(is_timeout_error(RuntimeError("service unavailable")))

    def test_append_only_local_jsonl_contains_audit_fields(self):
        result = {
            "raw_label": "YELLOW",
            "probabilities": [0.1, 0.7, 0.2],
            "rule_override": True,
            "matched_rule": "short of breath",
            "confidence_escalation": None,
            "label": "RED",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "decisions.jsonl"
            append_decision_event("I am short of breath", result, path)
            append_decision_event("I am short of breath", result, path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["input_text"], "I am short of breath")
        self.assertEqual(rows[0]["raw_prediction"], "YELLOW")
        self.assertEqual(rows[0]["confidence_scores"], {"RED": 0.1, "YELLOW": 0.7, "GREEN": 0.2})
        self.assertTrue(rows[0]["rule_override"])
        self.assertEqual(rows[0]["matched_rule"], "short of breath")
        self.assertEqual(rows[0]["final_classification"], "RED")
        self.assertTrue(rows[0]["timestamp_utc"].endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
