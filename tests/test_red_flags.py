"""Phrase-level smoke tests for the reviewed deterministic emergency rules."""
import unittest

from safety_rules import matched_red_flag


CASES = [
    ("chest pain", "crushing chest pain radiating to my jaw"),
    ("breathing", "I am gasping for air and struggling to breathe"),
    ("stroke", "his face is drooping and his speech is slurred"),
    ("anaphylaxis", "I have throat swelling after an allergic reaction"),
    ("severe bleeding", "there is spurting blood and blood soaking through the towel"),
    ("overdose", "I took too many pills and feel very drowsy"),
    ("suicidality", "I might hurt myself tonight"),
]

NEGATED_CASES = [
    ("chest pain", "I have no crushing chest pain"),
    ("breathing", "I am not gasping for air"),
    ("stroke", "my face was drooping last year, but it resolved"),
    ("anaphylaxis", "I have no throat swelling"),
    ("severe bleeding", "there is no spurting blood"),
    ("overdose", "I did not take too many pills"),
    ("suicidality", "I do not want to die"),
]


class RedFlagTests(unittest.TestCase):
    def test_category_phrases_trigger_red_rule(self):
        for category, text in CASES:
            with self.subTest(category=category):
                self.assertTrue(matched_red_flag(text), f"{category} phrase did not match")

    @unittest.skip("Negation scope is implemented in Fix 2")
    def test_negated_or_historical_phrase_does_not_trigger(self):
        for category, text in NEGATED_CASES:
            with self.subTest(category=category):
                self.assertIsNone(matched_red_flag(text), f"{category} context should not force RED")


if __name__ == "__main__":
    unittest.main()
