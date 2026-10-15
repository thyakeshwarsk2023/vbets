# 🚑 V-BETS: Voice-Based Emergency Triage System

V-BETS is an academic voice-enabled triage prototype combining Speech-to-Text (SpeechRecognition), Natural Language Processing, Deep Learning (Bag-of-Words MLP), and deterministic safety-rule overrides to prioritize medical emergencies into three triage levels:
- 🔴 **RED – Critical**: Immediate life threats (airway compromise, unconsciousness, cardiac arrest, severe hemorrhage).
- 🟡 **YELLOW – Urgent**: Urgent medical attention required (fractures, severe pain, moderate infection, deep lacerations).
- 🟢 **GREEN – General**: Non-urgent inquiries, routine appointments, prescription refills, and administrative questions.

---

## Quickstart

### 1. Local Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training & Evaluation
```bash
python train.py             # Trains BoW MLP & comparison model; outputs to model/
python evaluate_unseen.py   # Evaluates model + rule layer on independent unseen test set
```

### 3. Launch App
```bash
streamlit run app.py
```

---

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub (ensure `model/` artifacts are included).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **"New app"**.
3. Select:
   - **Repository**: `<your-github-username>/<repo-name>`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. In **Advanced settings**:
   - Set **Python version**: `3.11` (also pinned in `runtime.txt`).
5. Click **"Deploy!"**.
