"""Policy tests for symmetric low-confidence escalation."""
import unittest
import numpy as np

from triage import GREEN_CONF_THRESHOLD, YELLOW_CONF_THRESHOLD, classify_text


class FakeModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype="float32")

    def predict(self, _features, verbose=0):
        return self.probs[None, :]


class ConfidenceEscalationTests(unittest.TestCase):
    def test_low_confidence_yellow_escalates_to_red(self):
        result = classify_text("I have a sore throat", FakeModel([0.26, 0.40, 0.34]), {})
        self.assertEqual(result["raw_label"], "YELLOW")
        self.assertEqual(result["label"], "RED")
        self.assertEqual(result["confidence_escalation"], "low_confidence_yellow_to_red")
        self.assertIn("escalated to RED", result["note"])

    def test_low_confidence_green_escalates_to_yellow(self):
        result = classify_text("routine visit", FakeModel([0.30, 0.32, 0.38]), {})
        self.assertEqual(result["label"], "YELLOW")
        self.assertEqual(result["confidence_escalation"], "low_confidence_green_to_yellow")

    def test_thresholds_are_named_and_default_to_half(self):
        self.assertEqual(GREEN_CONF_THRESHOLD, 0.50)
        self.assertEqual(YELLOW_CONF_THRESHOLD, 0.50)

    def test_routine_refill_phrase_prevents_vocabulary_only_red(self):
        result = classify_text(
            "hello i need my monthly maintenance blood pressure medication refilled because my bottle has only two pills left",
            FakeModel([0.80, 0.10, 0.10]), {},
        )
        self.assertEqual(result["label"], "GREEN")
        self.assertEqual(result["matched_rule"], "routine_prescription_request")
        self.assertTrue(result["rule_override"])

    def test_negated_mild_gardening_back_complaint_is_not_escalated(self):
        result = classify_text(
            "i spent the weekend gardening and my lower back feels stiff and achy when i bend down but i have no shooting pain numbness or weakness in my legs",
            FakeModel([0.01, 0.97, 0.02]), {},
        )
        self.assertEqual(result["label"], "GREEN")
        self.assertEqual(result["matched_rule"], "negated_mild_gardening_back_complaint")
        self.assertTrue(result["rule_override"])


if __name__ == "__main__":
    unittest.main()
