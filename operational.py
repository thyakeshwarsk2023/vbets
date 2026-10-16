"""Local-only operational helpers for the Streamlit demo."""
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

from preprocess import LABELS, preprocess


def is_usable_transcript(text, minimum_tokens=2):
    """Require at least two normalized tokens before attempting classification."""
    return len(preprocess(text or "")) >= minimum_tokens


def is_timeout_error(error):
    """Recognize direct and SpeechRecognition-wrapped network timeouts."""
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    reason = getattr(error, "reason", None)
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return True
    return "timed out" in str(error).lower() or "timeout" in str(error).lower()


def build_decision_event(text, result):
    """Create the exact audit payload without writing it anywhere."""
    probabilities = result["probabilities"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Raw text is retained for this synthetic academic demo as requested.
        # Production deployments must hash/encrypt this field and define retention.
        "input_text": text,
        "raw_prediction": result["raw_label"],
        "confidence_scores": {
            label: float(probabilities[i]) for i, label in enumerate(LABELS)
        },
        "rule_override": bool(result["rule_override"]),
        "matched_rule": result["matched_rule"],
        "confidence_escalation": result["confidence_escalation"],
        "final_classification": result["label"],
    }


def append_decision_event(text, result, path="logs/decisions.jsonl"):
    """Append one JSON object per line to a local audit file."""
    event = build_decision_event(text, result)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    return target
