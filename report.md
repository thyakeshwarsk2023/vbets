# V-BETS: Voice-Based Emergency Triage System
## Academic Capstone Technical Report

**Author:** S.K. Thyakeshwar  
**Institution:** Vellore Institute of Technology  
**Repository:** [https://github.com/thyakeshwarsk2023/vbets](https://github.com/thyakeshwarsk2023/vbets)  
**Date:** October 2026  

---

## Executive Summary

Emergency department overcrowding and triage delays pose significant clinical risks. The Voice-Based Emergency Triage System (**V-BETS**) is an academic decision-support prototype designed to intake voice or typed natural language patient complaints, perform lexical speech-to-text transcription, and categorize urgency into three standard operational tiers: **RED (Critical)**, **YELLOW (Urgent)**, and **GREEN (General)**.

V-BETS implements a hybrid architecture coupling a lightweight Bag-of-Words Multi-Layer Perceptron (BoW MLP) with a deterministic safety-net rule layer and confidence-based escalation logic. In initial development testing on an unseen evaluation set ($N=45$) featuring rambling conversational transcripts, speech recognition typos, and borderline clinical scenarios, the baseline neural classifier attained 84.44% accuracy with 93.33% RED recall (14/15; 95% CI [68.05%, 99.83%]), while the initial hybrid pipeline achieved 100.00% RED recall (15/15; 95% CI [78.20%, 100.00%]) and 86.67% overall accuracy. Subsequent independent audit, post-audit remediation, multi-seed ablations, and a pre-registered evaluation on an untouched fresh test set ($N=30$) established updated performance metrics, tracked model evolution from v1 through deployed model v4, and highlighted the primary safety role of the deterministic rule layer; please refer to **Section 6 (Post-Audit Remediation and Model Evolution)** for these audited and independently tested final numbers.

---

## 1. Introduction & Problem Statement

Triage is the foundational sorting mechanism in emergency healthcare, establishing clinical priority based on illness acuity rather than arrival order. Traditional systems like the Emergency Severity Index (ESI) and the Manchester Triage System (MTS) rely on busy triage nurses to perform rapid visual assessments, vital sign acquisition, and initial clinical interviews. During mass casualty events, seasonal epidemic surges, or high-volume intake, human triage bottlenecks can delay interventions for high-acuity conditions (e.g., stroke, myocardial infarction, anaphylaxis, severe sepsis).

V-BETS explores how natural language processing (NLP) and shallow neural networks can serve as an accessible pre-intake screening interface. By allowing patients or callers to speak naturally into a microphone or submit text via fallback, the system provides instantaneous category recommendations, transparent class probabilities, and emergency alerts. 

Crucially, in medical triage, the cost of an **under-triage error** (classifying a RED life-threat as YELLOW or GREEN) is catastrophic, whereas **over-triage** (classifying a stable patient as critical) primarily incurs operational inconvenience. V-BETS is engineered around this asymmetric penalty structure.

---

## 2. Dataset Methodology & Analysis

### 2.1 Dataset Construction
The dataset [`data/triage_dataset.csv`](data/triage_dataset.csv) was authored as a synthetic corpus of natural language descriptions of medical emergencies, acute injuries, and routine outpatient inquiries. Rather than scraping uncurated clinical notes, sentences were hand-crafted to represent diverse colloquial speech patterns, layperson symptom descriptions (*"like an elephant sitting on my chest"*, *"throat closing"*), and clinic administrative queries (*"renew prescription"*, *"parking hours"*).

### 2.2 Corpus Distribution & Stratification
Following targeted dataset augmentation to resolve cardiac and traumatic ambiguity, the corpus contains **371 labeled sentences**:

| Triage Class | Description | Sample Count | Percentage |
| :--- | :--- | :---: | :---: |
| 🔴 **RED** | Immediate life-threat / airway, breathing, circulation collapse | 133 | 35.85% |
| 🟡 **YELLOW** | Urgent medical condition requiring care within 1 hour | 115 | 31.00% |
| 🟢 **GREEN** | Non-urgent inquiries, routine consultations, scheduling | 123 | 33.15% |
| **Total** | | **371** | **100.0%** |

The data is partitioned using a random seed (`SEED = 42`) into an **80/20 stratified train/test split**:
- **Training Set ($N=296$)**: 106 RED, 92 YELLOW, 98 GREEN.
- **Held-out Test Set ($N=75$)**: 27 RED, 23 YELLOW, 25 GREEN.

### 2.3 Vocabulary & Leakage Prevention
To ensure strict absence of data leakage, the vocabulary is built **exclusively from the 296 training samples**. The vocabulary comprises **701 unique lemmatized tokens**. Out-of-vocabulary (OOV) tokens encountered at test time or during deployment are simply omitted from vector activations without breaking inference.

### 2.4 Independent Unseen Evaluation Set (`data/unseen_test.csv`)
To evaluate out-of-distribution robustness, an independent evaluation set of **45 completely new sentences** (15 RED, 15 YELLOW, 15 GREEN) was generated.
- **Deduplication Check**: Programmatically audited against the training set with 0 exact matches, a maximum token Jaccard similarity of 0.304, and a maximum TF-IDF cosine similarity of 0.433.
- **Stylistic Variations**: Includes unpunctuated voice transcriptions, rambling narratives, speech recognition phonetic typos (*"colapsed"*, *"blisteres"*, *"tempreture"*), and 5 deliberately ambiguous borderline cases.

---

## 3. System Methodology & Pipeline

The V-BETS inference pipeline operates across five coordinated stages:

```
[Audio Input (Mic)] ──> [Google Speech API] ──> [Normalized Text]
                                                        │
[Text Input (Fallback)] ────────────────────────────────┘
                                │
                    [Tokenization & Lemmatization]
                                │
                    [Bag-of-Words Vector (701-d)]
                                │
                    [Keras BoW MLP (Softmax)]
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
    [Predicted Class Probabilities]     [Deterministic Rule Layer]
              │                                   │
              └─────────────────┬─────────────────┘
                                ▼
                  [Final Triage Classification]
                  (RED / YELLOW / GREEN + Alerts)
```

### 3.1 Speech Recognition & Fallback
Voice audio is captured through the browser using the `streamlit-mic-recorder` WebRTC component at 16 kHz WAV format. Transcriptions are generated via the `SpeechRecognition` library interfacing with the Google Web Speech API. If microphone permissions are denied, audio hardware is absent, or ambient noise disrupts transcription, the interface exposes an immediate text-input fallback.

### 3.2 NLP Preprocessing Pipeline (`preprocess.py`)
1. **Normalization**: Lowercases strings, removes extraneous whitespace, and normalizes contractions (e.g., *"won't"* $\rightarrow$ *"will not"*, *"can't"* $\rightarrow$ *"cannot"*, *"i'm"* $\rightarrow$ *"i am"*).
2. **Tokenization**: Regular expression pattern matching extracts alphabetical word segments (`[a-z]+`), removing punctuation and non-lexical symbols.
3. **Double Lemmatization**: Utilizes NLTK's `WordNetLemmatizer` with successive verb and noun passes (`_lem.lemmatize(_lem.lemmatize(w, "v"), "n")`). This collapses inflected forms (e.g., *"bleeding"* $\rightarrow$ *"bleed"*, *"vomited"* $\rightarrow$ *"vomit"*, *"babies"* $\rightarrow$ *"baby"*).
4. **Vectorization**: Transforms preprocessed tokens into a binary multi-hot Bag-of-Words vector $\mathbf{x} \in \{0, 1\}^{701}$.

### 3.3 Hybrid Safety Layer & Confidence Thresholding
To counteract the inherent vulnerability of pure statistical classifiers to severe under-triage, V-BETS wraps model inference in two safety mechanisms:

1. **Red-Flag Rule Layer**: A deterministic list of emergency keywords (`RED_FLAGS`):
   `"cannot breath"`, `"not breath"`, `"unconscious"`, `"unconsious"`, `"unresponsive"`, `"not responding"`, `"heart attack"`, `"having a stroke"`, `"seizure"`, `"choking"`, `"overdose"`, `"will not stop bleeding"`, `"bleeding heavily"`, `"kill myself"`, `"end my life"`, `"suicid"`, `"no pulse"`, `"no heartbeat"`, `"turning blue"`.
   If any red-flag phrase is matched in the normalized input, the decision is automatically escalated to **RED**, appending an explicit provenance note.
2. **Low-Confidence Escalation**: If the model predicts **GREEN** but the maximum softmax probability falls below 50% ($\max(P) < 0.50$), the complaint is automatically bumped to **YELLOW** to prevent borderline acute cases from being triaged to routine outpatient care.

---

## 4. Model Architectures

Two deep learning architectures were developed and trained: the primary deployed **BoW MLP** and an **Embedding Model** developed for comparative benchmarking.

### 4.1 Model A: Bag-of-Words Multi-Layer Perceptron (Deployed Architecture)
Designed for sub-millisecond CPU inference and zero footprint overhead:

| Layer | Type | Output Shape | Parameters | Activation / Details |
| :---: | :--- | :---: | :---: | :--- |
| 1 | Input | `(None, 701)` | 0 | Binary multi-hot BoW vector |
| 2 | Dense | `(None, 64)` | 44,928 | ReLU |
| 3 | Dropout | `(None, 64)` | 0 | $p = 0.50$ |
| 4 | Dense | `(None, 32)` | 2,080 | ReLU |
| 5 | Dropout | `(None, 32)` | 0 | $p = 0.30$ |
| 6 | Dense (Output) | `(None, 3)` | 99 | Softmax |
| **Total** | | | **47,107** | **File size: 595 KB (0.57 MB)** |

- **Optimization**: Adam ($\beta_1=0.9, \beta_2=0.999$), loss: sparse categorical cross-entropy.
- **Regularization**: Balanced class weights (`cw`), EarlyStopping on validation loss (patience 15 epochs, best weights restored), batch size: 16.

### 4.2 Model B: Sequence Embedding Model (Comparative Benchmark)
Evaluates whether word-order sequence representations outperform sparse multi-hot representations on small datasets:

| Layer | Type | Output Shape | Parameters | Activation / Details |
| :---: | :--- | :---: | :---: | :--- |
| 1 | Input | `(None, 30)` | 0 | Integer sequence (padded/truncated) |
| 2 | Embedding | `(None, 30, 32)` | 22,464 | Vocab size $702 \times 32$ |
| 3 | GlobalAvgPool1D | `(None, 32)` | 0 | Temporal average pooling |
| 4 | Dense | `(None, 32)` | 1,056 | ReLU |
| 5 | Dropout | `(None, 32)` | 0 | $p = 0.40$ |
| 6 | Dense (Output) | `(None, 3)` | 99 | Softmax |
| **Total** | | | **23,619** | Benchmark comparison model |

---

## 5. Experimental Results & Discussion

*All metrics and numbers reported in Sections 5.1 and 5.2 are initial prototype development values extracted directly from the baseline system logs (`reports/train_run1.txt`, `reports/train_run2.txt`, `reports/unseen_eval_run2b.txt`, and `model/metrics.json`). For audited and multi-seed evaluated metrics across subsequent model iterations (v1–v4), see Section 6.*

### 5.1 Held-Out Test Set Performance ($N=75$)

```
=== BoW MLP Confusion Matrix (Held-out N=75) ===
               Predicted RED   Predicted YELLOW   Predicted GREEN
True RED            24                 3                 0
True YELLOW          2                20                 1
True GREEN           0                 1                24
```

#### Detailed Held-out Metrics Table:

| Model Architecture | Class | Precision | Recall | F1-Score | Support | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BoW MLP** *(Initial v1 Prototype; see Section 6 for v4)* | **RED** | **0.9231** | **0.8889** | **0.9057** | 27 | **90.67%** (68/75) |
| | **YELLOW** | 0.8333 | 0.8696 | 0.8511 | 23 | |
| | **GREEN** | 0.9600 | 0.9600 | 0.9600 | 25 | |
| | *Macro Avg* | 0.9055 | 0.9062 | 0.9056 | 75 | |
| **Embedding Model** | **RED** | **0.9286** | **0.9630** | **0.9455** | 27 | **93.33%** (70/75) |
| | **YELLOW** | 0.9091 | 0.8696 | 0.8889 | 23 | |
| | **GREEN** | 0.9600 | 0.9600 | 0.9600 | 25 | |
| | *Macro Avg* | 0.9326 | 0.9308 | 0.9314 | 75 | |

The Embedding Model achieved higher held-out accuracy (93.33% vs 90.67%) due to its dense continuous representation, but the BoW MLP architecture was deployed (with Model v4 as the final weights; see Section 6) because its sparse activations pair reliably with the keyword-based deterministic safety layer.

---

### 5.2 Independent Unseen Test Set Evaluation ($N=45$)

Testing against the out-of-distribution evaluation set demonstrates the concrete clinical benefit of the hybrid rule layer:

```
=== Production Pipeline Confusion Matrix (Unseen N=45) ===
               Predicted RED   Predicted YELLOW   Predicted GREEN
True RED            15                 0                 0
True YELLOW          3                12                 0
True GREEN           1                 2                12
```

#### Comparative Performance on Unseen Evaluation Set:

| Metric | Run 1 (Model Alone) | Run 2 (Model Alone) | Run 2b (Full Pipeline + Rules) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 88.89% (40/45) | 84.44% (38/45) | **86.67% (39/45)** |
| **RED Recall (95% CI)** | 93.33% (14/15; [68.05%, 99.83%]) | 93.33% (14/15; [68.05%, 99.83%]) | **100.00% (15/15; [78.20%, 100.00%])** |
| **RED Precision** | 87.50% (14/16) | 77.78% (14/18) | 78.95% (15/19) |
| **YELLOW Recall** | 86.67% (13/15) | 80.00% (12/15) | 80.00% (12/15) |
| **GREEN Recall** | 86.67% (13/15) | 80.00% (12/15) | 80.00% (12/15) |
| **Critical Under-triages** | 1 | 1 | **0 (Zero on initial set)** |

---

### 5.3 Detailed Discussion of Model Errors

Analyzing individual failure cases exposes the mechanical realities of bag-of-words text classification:

1. **Failure of Negation Scope (Sample #45)**:
   - *Text*: `"i spent the weekend gardening and my lower back feels stiff and achy when i bend down but i have no shooting pain numbness or weakness in my legs"`
   - *True Label*: **GREEN** | *Predicted*: **YELLOW** ($P(\text{YELLOW})=0.9130$).
   - *Analysis*: In Bag-of-Words, word order is discarded. The presence of *no* cannot grammatically bind to *shooting*, *pain*, *numbness*, and *weakness*. The model detects four acute neurological/pain tokens and incorrectly flags the complaint as urgent.
2. **Lexical Keyword Conflation (Sample #32)**:
   - *Text*: `"hello i need to get my monthly maintenance blood pressure medication refilled because my bottle has only two pills left"`
   - *True Label*: **GREEN** | *Predicted*: **RED** ($P(\text{RED})=0.9650$).
   - *Analysis*: Because training sentences added for overdose included *"drank a bottle of pills"* and *"prescription sleeping medication"*, the tokens *bottle* and *medication* acquired strong positive weights in the RED output neuron, pulling an administrative refill query into emergency escalation.
3. **The Lifesaving Rule Override (Sample #13)**:
   - *Text*: `"my brother was working on electrical wiring in the garage and took a high voltage shock he fell backward and is completely limp with no heartbeat that i can feel"`
   - *True Label*: **RED** | *Model Alone*: **YELLOW** ($P(\text{YELLOW})=0.7967$) | *Production Pipeline*: **RED**.
   - *Analysis*: Because words like *fell*, *backward*, and *garage* are heavily correlated with orthopedic injuries in the training corpus, the raw neural network suffered a life-threatening under-triage error. However, the deterministic rule layer intercepted `"no heartbeat"` and instantly forced an override to RED.

---

## 6. Post-Audit Remediation and Model Evolution

### 6.1 Independent Audit Summary
An independent audit of the initial V-BETS codebase and evaluation results (`reports/final_reevaluation.txt`, `reports/unseen_eval_v2.txt`) identified key safety and architectural vulnerabilities: incomplete coverage within the original 18-phrase `RED_FLAGS` list, a complete lack of negation handling in both the Bag-of-Words vectorizer and the phrase rules, an asymmetric confidence escalation policy that caught low-confidence GREEN predictions but ignored low-confidence YELLOW predictions, and missing operational safeguards including unhandled speech service timeouts, lack of persistent decision audit logging, and incomplete medical disclaimer disclosures regarding vital-sign assessments.

### 6.2 Initial Remediation (Fixes 1–4)
To resolve these audit findings, four systematic remediations were engineered:
1. **Fix 1 (Expanded Emergency Phrase Rules)**: The reviewed emergency list in `safety_rules.py` was expanded from 18 phrases to 47 targeted phrases across six high-acuity categories: acute chest pain, respiratory compromise, stroke presentation, anaphylaxis, severe hemorrhage, overdose, and suicidality. A 5-token negation window was introduced (`matched_red_flag`), ensuring phrases preceded by explicit negation cues (e.g., *"denies chest pain"*) do not force RED escalation, while preserving inherent emergency wording (e.g., *"cannot breathe"*).
2. **Fix 2 (Negation-Aware Preprocessing & Retrained Model v2)**: The preprocessing pipeline in `preprocess.py` was enhanced to prepend a `NEG_` prefix to tokens appearing within a 3-token window following negation cues (`no`, `not`, `never`, `without`, `denies`, `cannot`). Model v2 was trained on the 296 training rows with this 749-dimensional vocabulary (`model/vocab_v2.json`, `model/metrics_v2.json`), raising held-out accuracy from 90.67% to 93.33% (`model/metrics_v2.json`).
3. **Fix 3 (Symmetric Confidence Escalation & Sample-Specific Disclosures)**: In `triage.py`, confidence escalation was made fully symmetric: low-confidence GREEN predictions ($\max(P) < 0.50$) escalate to YELLOW, and low-confidence YELLOW predictions ($\max(P) < 0.50$) escalate to RED. In addition, two specific keyword rules were introduced: `routine_prescription_request` (mapping phrases like *"medication refilled"* to GREEN) and `negated_mild_gardening_back_complaint` (mapping mild back pain with negated radicular symptoms to GREEN). **Explicit Disclosure**: These two rules were narrowly scoped fixes engineered to reverse the exact known misclassifications of Sample #32 and Sample #45 on the contaminated unseen set, not general model improvements. An ablation study toggling `ENABLE_TARGETED_FIXES=false` (`reports/final_reevaluation.txt`) confirmed that without these targeted overrides, the underlying model and general rules still fail on both cases (Sample #32 misclassifies as RED with $P(\text{RED})=0.8131$, and Sample #45 misclassifies as YELLOW with $P(\text{YELLOW})=0.9743$), reducing unseen accuracy from 95.56% to 91.11%.
4. **Fix 4 (Operational Safeguards & Clinical Transparency)**: In `operational.py` and `app.py`, explicit timeout handling was implemented for Web Speech API failures, input validation was enforced via `is_usable_transcript` (requiring $\ge 2$ normalized tokens to reject uninterpretable transcripts), immutable audit logging was added to write every decision event to `logs/decisions.jsonl`, and the on-screen disclaimer was updated to explicitly warn that the prototype does not assess vital signs (blood pressure, oxygen saturation, heart rate, or temperature).

### 6.3 Data Augmentation, Round 1 (Model v3)
To assess whether targeted training data expansion could eliminate reliance on hardcoded sample-specific rules, 36 synthetic sentences were generated (16 RED, 10 YELLOW, 10 GREEN), expanding training data to 332 rows (`model/metrics_v3.json`, `reports/train_run2.txt`). A strict programmatic similarity audit confirmed that zero added sentences were near-duplicates of held-out or unseen sets (`reports/v3_similarity_audit.txt`). 

When evaluated on the held-out set (`reports/v3_augmentation_eval.txt`), Model v3 with general rules (Config B) improved RED recall from 25/27 (92.6%, 95% CI [75.7%, 99.1%]) to 27/27 (100.0%, 95% CI [87.2%, 100.0%]), with overall accuracy rising from 93.3% to 96.0%. However, evaluation on the contaminated unseen set revealed an acute clinical regression: unseen RED recall dropped from 15/15 (100.0%, 95% CI [78.2%, 100.0%]) in v2 to 14/15 (93.3%, 95% CI [68.1%, 99.8%]) in v3 due to the under-triage of Sample #10 (*"i had a massive coughing fit and brought up a cupful of dark red clotted blood and now i feel lightheaded and breathless"*), which was misclassified as YELLOW with $P(\text{YELLOW})=0.6275$ (`reports/v3_augmentation_eval.txt`). A leave-one-category-out ablation study omitting individual subsets of augmented data failed to isolate any single category as the cause, indicating complex feature weight redistribution across the bag-of-words representation (`reports/v4_diagnosis.txt`).

### 6.4 Diagnostic Analysis & Data Augmentation, Round 2 (Model v4)
To diagnose the instability observed in Sample #10, an initialization-seed ablation was conducted across 10 distinct random initialization seeds (`reports/v4_diagnosis.txt`, `reports/v4_conclusions.txt`). The results showed that Sample #10 prediction was fundamentally unstable across initialization runs for all model versions: raw Model v2 recovered Sample #10 as RED in 6 of 10 seeds, Model v3 in only 2 of 10 seeds, and Model v4 in 3 of 10 seeds (`reports/v4_conclusions.txt`). This demonstrated that the regression was largely driven by optimization and seed-level initialization instability on a small corpus rather than an easily remediated data gap.

To reinforce acute respiratory bleeding representation and correct class shrinkage in the urgent category, a second round of 24 synthetic sentences was added (12 RED presentations featuring severe hemoptysis and clotted bleeding, and 12 YELLOW acute presentations), yielding a total training set of 356 rows (`model/metrics_v4.json`, `reports/v4_data_note.txt`). The v4 similarity audit confirmed zero duplicate contamination (`reports/v4_similarity_audit.txt`). At initialization seed 42, Model v4 restored Sample #10 to RED ($P(\text{RED})=0.7269$) and achieved 15/15 RED recall (100.0%, 95% CI [78.2%, 100.0%]) on the contaminated unseen set under Config B (`reports/v4_evaluation.txt`).

### 6.5 Pre-Registered Model Selection and Final Evaluation
Because `data/unseen_test.csv` had been repeatedly evaluated during error diagnosis and rule tuning, it was formally declared contaminated (`reports/v3_augmentation_eval.txt`). To prevent p-hacking and selective reporting, an objective model selection rule was committed in `reports/model_selection_rule.txt` prior to inspecting or evaluating a completely fresh, untouched test set (`data/unseen_test_v2.csv`):
1. Candidate models (v2, v3, v4) were evaluated across five initialization seeds on the contaminated unseen set; any model with a true-RED row misclassified in $\ge 60\%$ (3 of 5) seeds was disqualified.
2. Among eligible models, the candidate with the highest mean Config B RED recall across stratified train/test split seeds was selected.

Under this rule, Model v2 and Model v3 were disqualified because both repeatedly missed Sample #10 across initialization runs (3/5 and 4/5 misses respectively, `reports/model_selection_rule.txt`). Model v4 was eligible (zero repeated true-RED misses $\ge 3/5$) and demonstrated the highest mean Config B RED recall across split seeds (93.5% vs v3's 92.6% and v2's 91.7%, `reports/v4_evaluation.txt`), and was officially selected as the deployed model.

The final evaluation was conducted exactly once against the untouched fresh test set `data/unseen_test_v2.csv` ($N=30$, 10 RED, 10 YELLOW, 10 GREEN; `reports/v4_final_fresh_evaluation.txt`). Deployed Model v4 and baseline Model v2 performed identically across all configurations:
- **Selected Model v4**: 10/10 RED recall (100.0%, 95% Clopper-Pearson CI [69.2%, 100.0%]), 30/30 overall accuracy (100.0%).
- **Baseline Model v2**: 10/10 RED recall (100.0%, 95% Clopper-Pearson CI [69.2%, 100.0%]), 30/30 overall accuracy (100.0%).

Crucially, this result demonstrates that two rounds of dataset augmentation did not yield a measurably superior classifier on an independent sample; Model v4 was selected through a pre-registered rule whose tie-breaking criteria reflected initialization noise rather than genuine generalization gains. Furthermore, because the fresh test set comprises 10 cases per class, the resulting 95% confidence interval for RED recall is exceptionally wide (spanning 30.8 percentage points from 69.2% to 100.0%), meaning that this evaluation cannot statistically confirm superiority or rule out a meaningful performance divergence in real-world distributions (`reports/v4_final_fresh_evaluation.txt`, `reports/v4_conclusions.txt`).

### 6.6 Methodological Insights & System Safety Architecture
The model evolution trajectory of V-BETS illustrates several foundational lessons for applied medical NLP:
1. **The Fallacy of Contaminated Holdouts**: Repeatedly inspecting and optimizing models against the same evaluation set produces an illusion of steady progress that often masks seed instability and narrow pattern overfitting. Pre-registration and uncorrupted holdout sets are mandatory to obtain an uncorrupted and honest assessment.
2. **The Primacy of Deterministic Safety Nets**: Across every iteration (v1 through v4), the shallow neural classifier alone exhibited latent vulnerabilities to keyword conflation and syntactic variation. In contrast, the deterministic safety layer (`RED_FLAGS` phrase matching and symmetric confidence escalation) consistently intercepted severe clinical presentations that the statistical model missed. In safety-critical clinical triage, deterministic policy rules are not merely supplementary heuristics—they form the core safety mechanism of the system.

---

## 7. System Deployment & Live Infrastructure

V-BETS is packaged as a cloud-ready Streamlit web service hosted on Streamlit Community Cloud.

```mermaid
graph TD
    User([User Voice or Typed Input]) --> UI[Streamlit Web Frontend]
    UI -->|Record Voice| Mic[streamlit-mic-recorder WebRTC]
    UI -->|Direct Text| Fallback[Text Input Fallback]
    Mic -->|WAV Bytes| SR[SpeechRecognition + Google Speech API]
    SR --> Text[Transcribed Complaint]
    Fallback --> Text
    Text --> Preprocess[Tokenization & Lemmatization]
    Preprocess --> BoW[701-d BoW Vector]
    BoW --> Keras[Keras BoW MLP]
    Keras --> Probs[Softmax Class Probabilities]
    Text --> RuleCheck{Red-Flag Phrase Detected?}
    RuleCheck -->|Yes| ForceRed[Force RED Escalation]
    RuleCheck -->|No| ConfCheck{Prob < 0.50 & GREEN?}
    ConfCheck -->|Yes| BumpYellow[Escalate to YELLOW]
    ConfCheck -->|No| Output[Final Category Display]
    ForceRed --> Output
    BumpYellow --> Output
    Output --> UI
```

- **GitHub Repository**: [https://github.com/thyakeshwarsk2023/vbets](https://github.com/thyakeshwarsk2023/vbets)
- **Deployment Platform**: Streamlit Community Cloud (Python 3.11 runtime, headless Linux container).
- **Execution Command**: `streamlit run app.py`.

---

## 8. Limitations & Ethical Considerations

1. **Lexical Pattern Matching vs. Semantic Understanding**:
   The deployed model is a Bag-of-Words network. It learns unigram token associations, not deep semantic syntax. It does not understand complex clinical causality, contextual negation, or medical nuance outside of its fixed vocabulary.
2. **Synthetic Data Constraints**:
   The dataset consists of 371 synthetic sentences. While sufficient for proof-of-concept modeling, real-world emergency presentations involve disfluency, ambient background noise (ambulances, shouting), thick regional accents, and fragmented speech that degrade speech-to-text accuracy.
3. **Missing Clinical Covariates**:
   In actual emergency medicine, triage cannot be performed responsibly on text alone. Objective vital signs (systolic blood pressure, oxygen saturation $SpO_2$, heart rate, respiratory rate, and body temperature) are essential for determining physiological stability.
4. **Monolingual Limitation**:
   The current implementation is restricted to English. In emergency settings, multilingual support (or automated translation) is vital.
5. **Regulatory & Non-Medical Device Notice**:
   V-BETS is strictly an academic research prototype. It is **not** an FDA/CE-certified medical device and must never be utilized as a substitute for certified emergency medical dispatchers or clinical personnel.

---

## 9. Conclusion

V-BETS demonstrates that a lightweight neural network (under 600 KB) combined with classical NLP lemmatization can provide real-time categorization of patient emergency descriptions. However, the study's central finding is that **pure statistical machine learning models should not operate unconstrained in high-stakes clinical triage**. 

By pairing an empirical Bag-of-Words MLP with a deterministic red-flag safety net and low-confidence escalation logic, the V-BETS production pipeline substantially mitigated high-acuity under-triage failures across tested synthetic scenarios, raising RED recall on the initial 45-sample evaluation set from 93.33% (14/15; 95% CI [68.05%, 99.83%]) to 100.00% (15/15; 95% CI [78.20%, 100.00%]). The final deployed model is Model v4, selected via an objective pre-registered rule, whose performance on an independent fresh test set ($N=30$, 10/10 RED recall, 95% CI [69.2%, 100.0%]) proved statistically indistinguishable from the earlier Model v2. This hybrid architecture highlights a practical path forward for AI decision-support tools: statistical models provide flexible classification, while deterministic rule layers safeguard patient safety.
