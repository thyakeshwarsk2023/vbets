"""Regression checks for negated BoW features and inherent emergency phrases."""
import unittest

from preprocess import bow_vector, build_vocab, preprocess
from safety_rules import matched_red_flag


class NegationTests(unittest.TestCase):
    def test_windowed_negation_features_and_ab_switch(self):
        text = "I have no shooting pain numbness or weakness in my legs"
        tokens = preprocess(text, use_negation=True)
        self.assertIn("NEG_pain", tokens)
        self.assertIn("NEG_numbness", tokens)
        self.assertIn("NEG_weakness", tokens)
        self.assertNotIn("NEG_pain", preprocess(text, use_negation=False))

    def test_vocab_and_vectorizer_share_negation_setting(self):
        texts = ["no chest pain", "severe chest pain"]
        vocab = build_vocab(texts, use_negation=True)
        self.assertIn("NEG_pain", vocab)
        vector = bow_vector(texts[0], vocab, use_negation=True)
        self.assertEqual(vector[vocab["NEG_pain"]], 1.0)

    def test_inherent_emergency_negation_phrases_still_trigger(self):
        for text in ("he has no heartbeat", "she has no pulse", "the patient is not breathing", "he is not responding"):
            with self.subTest(text=text):
                self.assertTrue(matched_red_flag(text))

    def test_sample_45_symptoms_are_negated_features(self):
        text = "my lower back feels stiff and achy but i have no shooting pain numbness or weakness in my legs"
        tokens = preprocess(text)
        self.assertIn("NEG_pain", tokens)
        self.assertIn("NEG_numbness", tokens)
        self.assertIn("NEG_weakness", tokens)


if __name__ == "__main__":
    unittest.main()
