import io, json, socket
import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from tensorflow import keras
from preprocess import LABELS
from operational import append_decision_event, is_timeout_error, is_usable_transcript
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


def transcribe(wav_bytes, timeout_seconds=10):
    r = sr.Recognizer()
    # SpeechRecognition applies this bound to its Google Web Speech API request.
    r.operation_timeout = timeout_seconds
    with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
        audio = r.record(src)
    return r.recognize_google(audio)


model, vocab = load_assets()

st.title("🚑 V-BETS: Voice-Based Emergency Triage System")
st.warning("**Academic demo only.** This student prototype is trained on a small synthetic "
           "dataset. It is NOT a medical device and does NOT assess vital signs (blood pressure, "
           "oxygen saturation, heart rate, or temperature), and is NOT a substitute for professional "
           "care or in-person vital-sign assessment. In a real emergency, call your local emergency number immediately.")

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
    except (TimeoutError, socket.timeout):
        st.error("Transcription timed out. Please type your complaint instead.")
    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please try again or type your message.")
    except sr.RequestError as e:
        if is_timeout_error(e):
            st.error("Transcription timed out. Please type your complaint instead.")
        else:
            st.error(f"Speech service unavailable ({e}). Please use the text box.")
    except (ValueError, OSError, EOFError) as e:
        st.error(f"Could not read the audio ({e}). Please try again or type your complaint.")
elif go and typed.strip():
    text = typed.strip()

if text is not None:
    if not is_usable_transcript(text):
        st.error("Could not understand. Please try again or type your complaint.")
    else:
        result = classify(text, model, vocab)
        label, probs, note = result["label"], result["probabilities"], result["note"]
        try:
            append_decision_event(text, result)
        except OSError as e:
            st.warning(f"Decision could not be written to the local audit log ({e}).")
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
