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

ALL_NEW_PHRASES = {
    "chest pain": ["crushing chest pain", "chest pressure", "pain in my jaw", "pain down my arm",
                   "chest feels tight", "elephant on my chest", "squeezing in my chest"],
    "breathing": ["short of breath", "gasping for air", "struggling to breathe", "throat is closing",
                  "throat closing", "can't get air", "cant get air", "difficulty breathing"],
    "stroke": ["face drooping", "one side is weak", "slurred speech", "sudden confusion",
               "sudden vision loss", "can't move one side", "face is drooping"],
    "anaphylaxis": ["throat swelling", "tongue swelling", "lips swelling",
                    "allergic reaction and trouble breathing", "hives and dizziness"],
    "severe bleeding": ["spurting blood", "blood soaking through", "won't clot", "wont clot",
                        "can't control the bleeding", "cant control the bleeding"],
    "overdose": ["took too many pills", "swallowed a handful", "ingested the whole bottle",
                 "took the whole bottle"],
    "suicidality": ["want to die", "might hurt myself", "planning to end it", "going to hurt myself"],
}

NEGATED_CASES = [
    ("chest pain", "I have no crushing chest pain"),
    ("breathing", "I am not gasping for air"),
    ("stroke", "There is no face drooping now; it happened last year but resolved."),
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

    def test_every_added_phrase_triggers_and_is_negation_scoped(self):
        for category, phrases in ALL_NEW_PHRASES.items():
            for phrase in phrases:
                with self.subTest(category=category, phrase=phrase):
                    self.assertTrue(matched_red_flag(f"Patient reports: {phrase}."))
                    self.assertIsNone(matched_red_flag(f"Patient denies: {phrase}."))

    def test_negated_or_historical_phrase_does_not_trigger(self):
        for category, text in NEGATED_CASES:
            with self.subTest(category=category):
                self.assertIsNone(matched_red_flag(text), f"{category} context should not force RED")


if __name__ == "__main__":
    unittest.main()
