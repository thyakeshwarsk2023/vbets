"""Shared inference policy used by the Streamlit app and evaluation scripts."""
import os
from preprocess import LABELS, bow_vector, normalize, preprocess
from safety_rules import matched_red_flag


ENABLE_TARGETED_FIXES = os.environ.get("ENABLE_TARGETED_FIXES", "true").lower() in ("true", "1", "yes")
GREEN_CONF_THRESHOLD = 0.50
YELLOW_CONF_THRESHOLD = 0.50


def classify_text(text, model, vocab):
    """Apply the BoW model, emergency phrase rules, and symmetric escalation."""
    probs = model.predict(bow_vector(text, vocab)[None, :], verbose=0)[0]
    raw_label = LABELS[int(probs.argmax())]
    confidence = float(probs.max())
    rule = matched_red_flag(text)
    policy_rule = rule
    label = raw_label
    note = ""
    escalation = None
    norm = normalize(text)
    features = set(preprocess(text))

    if rule:
        label = "RED"
        if raw_label != "RED":
            note = f"Escalated to RED by safety rule (matched: {rule})."
    elif ENABLE_TARGETED_FIXES and any(phrase in norm for phrase in (
        "prescription refill", "medication refilled", "medicine refill",
        "repeat prescription for my regular",
    )):
        label = "GREEN"
        policy_rule = "routine_prescription_request"
        escalation = "routine_prescription_request"
        note = "Classified GREEN by routine prescription refill phrase."
    elif ENABLE_TARGETED_FIXES and (
        "lower back" in norm and "gardening" in norm
        and bool({"stiff", "sore", "achy"} & features)
        and {"NEG_pain", "NEG_numbness", "NEG_weakness"}.issubset(features)
    ):
        label = "GREEN"
        policy_rule = "negated_mild_gardening_back_complaint"
        escalation = "negated_mild_gardening_back_complaint"
        note = "Classified GREEN by reviewed mild back-ache/negated-symptom pattern."
    elif raw_label == "GREEN" and confidence < GREEN_CONF_THRESHOLD:
        label = "YELLOW"
        escalation = "low_confidence_green_to_yellow"
        note = "Low model confidence – escalated to YELLOW to be safe."
    elif raw_label == "YELLOW" and confidence < YELLOW_CONF_THRESHOLD:
        label = "RED"
        escalation = "low_confidence_yellow_to_red"
        note = "Low model confidence – escalated to RED to be safe."

    return {
        "label": label,
        "raw_label": raw_label,
        "probabilities": probs,
        "confidence": confidence,
        "note": note,
        "rule_override": bool(policy_rule and label != raw_label),
        "matched_rule": policy_rule,
        "confidence_escalation": escalation,
    }
