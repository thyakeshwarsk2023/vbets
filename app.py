import io, json, socket
import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from tensorflow import keras
from preprocess import LABELS
from operational import append_decision_event, is_timeout_error, is_usable_transcript
from safety_rules import RED_FLAGS
from triage import classify_text

st.set_page_config(page_title="V-BETS Emergency Triage", layout="centered")

RESPONSES = {
    "RED": ("RED – CRITICAL", "#c62828",
            "This sounds life-threatening. Dispatching ambulance / alerting emergency team now.",
            "Clinical directive: Call local emergency services immediately or initiate immediate emergency resuscitation."),
    "YELLOW": ("YELLOW – URGENT", "#f9a825",
               "This needs prompt medical attention. Please proceed to the Urgent Care desk.",
               "Estimated priority: Clinician evaluation required within the hour."),
    "GREEN": ("GREEN – GENERAL", "#2e7d32",
              "This looks non-urgent. We will help you schedule an appointment or answer your query.",
              "Estimated priority: Non-urgent inquiry or routine outpatient consultation."),
}


@st.cache_resource
def load_assets():
    model = keras.models.load_model("model/triage_mlp_v4.keras")
    vocab = json.load(open("model/vocab_v4.json"))
    return model, vocab


def classify(text, model, vocab):
    return classify_text(text, model, vocab)


def transcribe(wav_bytes, timeout_seconds=10):
    r = sr.Recognizer()
    r.operation_timeout = timeout_seconds
    with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
        audio = r.record(src)
    return r.recognize_google(audio)


model, vocab = load_assets()

st.title("V-BETS: Voice-Based Emergency Triage System")
st.caption("Clinical Decision Support Prototype • Deployed Model: v4 (Augmented BoW MLP + Deterministic Safety Policy)")

st.warning("**Academic demo only.** This student prototype is trained on a small synthetic "
           "dataset. It is NOT a medical device and does NOT assess vital signs (blood pressure, "
           "oxygen saturation, heart rate, or temperature), and is NOT a substitute for professional "
           "care or in-person vital-sign assessment. In a real emergency, call your local emergency number immediately.")

st.subheader("1. Describe the patient's condition")
audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop & analyse",
                     format="wav", key="recorder")
typed = st.text_input("Or enter text description (fallback if microphone is unavailable):",
                      placeholder="e.g. Patient is experiencing sudden shortness of breath and chest pressure...")
go = st.button("Analyse complaint", type="primary")

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
        anim = "animation: pulse-border 2s infinite;" if label == "RED" else ""
        st.markdown(f"""
        <style>
          .stApp {{ background: {color}14; }}
          @keyframes pulse-border {{
            0% {{ box-shadow: 0 0 0 0 {color}88; }}
            70% {{ box-shadow: 0 0 0 10px {color}00; }}
            100% {{ box-shadow: 0 0 0 0 {color}00; }}
          }}
        </style>
        <div style="background:{color};color:white;padding:18px 24px;border-radius:10px;margin-bottom:14px;{anim}">
          <div style="font-size:0.8rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;opacity:0.9;">Triage Classification</div>
          <h2 style="margin:2px 0 0 0;color:white;font-size:1.7rem;">{title}</h2>
        </div>""", unsafe_allow_html=True)
        st.subheader("2. Result & Clinical Detail")
        st.markdown(f"**Intake Transcript:** {text}")
        st.markdown(f"**Recommended Action:** {message}")
        st.caption(action)
        if note:
            st.info(f"Policy Note: {note}")

        st.markdown("**Model Class Probabilities:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("RED (Critical)", f"{float(probs[0])*100:.1f}%")
        col2.metric("YELLOW (Urgent)", f"{float(probs[1])*100:.1f}%")
        col3.metric("GREEN (General)", f"{float(probs[2])*100:.1f}%")
        st.bar_chart({l: float(p) for l, p in zip(LABELS, probs)})
