"""Reviewed phrase rules for emergency escalation."""
from preprocess import normalize


# Keep additions phrase-specific; standalone tokens create avoidable false alarms.
RED_FLAGS = [
    # Existing phrases
    "cannot breath", "not breath", "unconscious", "unconsious", "unresponsive",
    "not responding", "heart attack", "having a stroke", "seizure", "choking",
    "overdose", "will not stop bleeding", "bleeding heavily", "kill myself",
    "end my life", "suicid", "no pulse", "no heartbeat", "turning blue",
    # Chest pain
    "crushing chest pain", "chest pressure", "pain in my jaw", "pain down my arm",
    "chest feels tight", "elephant on my chest", "squeezing in my chest",
    # Breathing difficulty
    "short of breath", "gasping for air", "struggling to breathe", "throat is closing",
    "throat closing", "can't get air", "cant get air", "difficulty breathing",
    # Stroke
    "face drooping", "one side is weak", "slurred speech", "sudden confusion",
    "sudden vision loss", "can't move one side", "face is drooping",
    # Anaphylaxis
    "throat swelling", "tongue swelling", "lips swelling",
    "allergic reaction and trouble breathing", "hives and dizziness",
    # Severe bleeding
    "spurting blood", "blood soaking through", "won't clot", "wont clot",
    "can't control the bleeding", "cant control the bleeding",
    # Overdose
    "took too many pills", "swallowed a handful", "ingested the whole bottle",
    "took the whole bottle",
    # Suicidality
    "want to die", "might hurt myself", "planning to end it", "going to hurt myself",
]


def matched_red_flag(text):
    """Return the first configured red-flag phrase found in normalized text."""
    norm = normalize(text)
    return next((flag for flag in RED_FLAGS if normalize(flag) in norm), None)
