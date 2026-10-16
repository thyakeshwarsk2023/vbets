import io, json
import numpy as np
import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from tensorflow import keras
from preprocess import LABELS, preprocess
from safety_rules import RED_FLAGS, matched_red_flag
from triage import classify_text

st.set_page_config(page_title="V-BETS Triage Demo", page_icon="🚑", layout="centered")

RESPONSES = {
    "RED": ("🔴 RED – CRITICAL", "#c62828",
            "This sounds life-threatening. Dispatching ambulance / alerting emergency team now.",
            "In real life: call your local emergency number immediately."),
    "YELLOW": ("🟡 YELLOW – URGENT", "#f9a825",
               "This needs prompt medical attention. Please proceed to the Urgent Care desk.",
               "Estimated priority: seen within the hour."),
    "GREEN": ("🟢 GREEN – GENERAL", "#2e7d32",
              "This looks non-urgent. We will help you schedule an appointment or answer your query.",
              "Please see the front desk or book online."),
}


@st.cache_resource
def load_assets():
    model = keras.models.load_model("model/triage_mlp_v2.keras")
    vocab = json.load(open("model/vocab_v2.json"))
    return model, vocab


def classify(text, model, vocab):
    return classify_text(text, model, vocab)


def transcribe(wav_bytes):
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
        audio = r.record(src)
    return r.recognize_google(audio)


model, vocab = load_assets()

st.title("🚑 V-BETS: Voice-Based Emergency Triage System")
st.warning("**Academic demo only.** This is a student prototype trained on a small synthetic "
           "dataset. It is NOT a medical device and NOT a substitute for professional care. "
           "In a real emergency, call your local emergency number immediately.")

st.subheader("1. Describe the situation")
audio = mic_recorder(start_prompt="🎙️ Start recording", stop_prompt="⏹️ Stop & analyse",
                     format="wav", key="recorder")
typed = st.text_input("…or type it here (fallback if the microphone fails)")
go = st.button("Analyse typed text")

text = None
if audio and audio.get("id") != st.session_state.get("last_audio_id"):
    st.session_state["last_audio_id"] = audio.get("id")
    try:
        text = transcribe(audio["bytes"])
    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please try again or type your message.")
    except sr.RequestError as e:
        st.error(f"Speech service unavailable ({e}). Please use the text box.")
elif go and typed.strip():
    text = typed.strip()

if text:
    result = classify(text, model, vocab)
    label, probs, note = result["label"], result["probabilities"], result["note"]
    title, color, message, action = RESPONSES[label]
    anim = "animation: blink 1s infinite;" if label == "RED" else ""
    st.markdown(f"""
    <style>
      .stApp {{ background: {color}22; }}
      @keyframes blink {{ 50% {{ opacity: 0.55; }} }}
    </style>
    <div style="background:{color};color:white;padding:22px;border-radius:12px;{anim}">
      <h2 style="margin:0">{title}</h2></div>""", unsafe_allow_html=True)
    st.subheader("2. Result")
    st.markdown(f"**Recognized speech:** {text}")
    st.markdown(f"**System response:** {message}")
    st.caption(action)
    if note:
        st.info(note)
    st.bar_chart({l: float(p) for l, p in zip(LABELS, probs)})
