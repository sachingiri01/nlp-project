# Hindi Legal Bail Outcome Classification

A decision-support system that predicts whether a Hindi bail application will be **GRANTED** or **DENIED**, shows the most similar past cases, and extracts the key legal facts from the case text. It uses classical NLP and ML only (TF-IDF, a linear SVM, cosine similarity and rule-based extraction). **No LLM is used**, so every output can be traced back to the data.

> This is a decision-support tool, not legal advice.

---

## Problem Statement

Our project uses Natural Language Processing and Machine Learning to classify Hindi legal bail documents as either GRANTED or DENIED. Since legal documents contain large amounts of information, manually analyzing them is time-consuming. We identify patterns in the facts and arguments presented in the case and use them to predict the bail outcome.

We deliberately exclude the judge's opinion and final decision, because including them would cause data leakage and let the model simply learn the answer instead of understanding the case.

---

## What the System Outputs

Given the facts of a case, the system produces three things.

### 1. Prediction with confidence

```
Predicted Bail Outcome:

🟢 GRANTED

Confidence: 84%
```

If confidence is below 60%, the system adds a "manual review recommended" warning.

### 2. Relevant historical cases with matching sentences

```
Relevant Historical Cases

1. Case #1823
   Similarity: 91%
   Outcome: GRANTED
   • Your case : **अभियुक्त** का **आपराधिक** **इतिहास** नहीं है।
     Past case : **अभियुक्त** का कोई **आपराधिक** **इतिहास** नहीं है।
```

The words shared between your case and the retrieved case are shown in `**bold**`, so it is clear why a case was retrieved.

### 3. Extracted key facts

```
Sections                    : 302, 307
Offence type                : Murder, Attempt to murder
Case crime no.              : 100/2020
Police station              : कोतवाली
Accused in custody          : Yes
Custody / arrest date       : 12.03.2020
Chargesheet                 : Filed
Prior record                : No prior criminal record stated
Claims false implication    : Yes
Co-accused already on bail  : Not mentioned
...
```

---

## Dataset

The project uses the **BAIL task from IL-TUR**, based on the **Hindi Legal Documents Corpus (HLDC)**.

**Dataset:** <https://huggingface.co/datasets/Exploration-Lab/IL-TUR/viewer/bail/train_all?p=1>

HLDC is a large Hindi legal dataset created from district court documents of Uttar Pradesh, India. The original corpus contains 912,568 Hindi legal documents collected from the e-Courts website for cases from May 2019 to May 2021.

For the bail task, the researchers extracted 340,280 bail-related documents. These were structured into sections such as facts and arguments and judge opinion, with the final bail decision labelled GRANTED or DENIED.

---

## Project Structure

| File | Purpose |
|---|---|
| `download_bail_data.py` | Downloads the bail dataset |
| `preprocess.py` | Unicode normalization and text cleaning |
| `baseline_tfidf.py` | Trains and evaluates the TF-IDF + calibrated Linear SVM model |
| `build_case_index.py` | Builds the searchable index of historical cases from the training data |
| `bail_engine.py` | Prediction, confidence, similar-case retrieval and sentence matching |
| `legal_facts.py` | Rule-based extraction of key facts from Hindi text |
| `predict.py` | Command-line interface for the full system |
| `model_tfidf_facts_only.joblib` | Trained model |
| `case_index.joblib` | Case index (created by `build_case_index.py`) |

---

## Pipeline

```
IL-TUR / HLDC Dataset
        ↓
Extract Facts & Arguments
        ↓
Remove Missing / Invalid Text
        ↓
Unicode Normalization + Noise Cleaning
        ↓
Train / Test Split
        ↓
TF-IDF Vectorization
        ↓
Calibrated Linear SVM
        ↓
┌───────────────────────┬──────────────────────────┬─────────────────────┐
│ Prediction + Confidence│ Similar Cases (cosine    │ Key-Fact Extraction │
│ GRANTED / DENIED       │ similarity on TF-IDF) +  │ (Hindi regex rules) │
│                        │ matching sentences       │                     │
└───────────────────────┴──────────────────────────┴─────────────────────┘
```

### 1. Text Extraction
Only `facts-and-arguments` is used as model input. `judge-opinion` is excluded to prevent label leakage.

### 2. Preprocessing
- Unicode NFC normalization
- Noise removal
- Whitespace normalization
- Stopword removal disabled

### 3. TF-IDF
The cleaned Hindi text is converted into numeric features using TF-IDF (unigrams and bigrams, up to 50,000 features). The vectorizer is fitted only on the training data.

### 4. Classification
A **Linear SVM** (`LinearSVC`, class-balanced) wrapped in `CalibratedClassifierCV`. Calibration turns the SVM scores into real probabilities, which are used as the confidence value. Logistic Regression is also available with `--model logreg`.

### 5. Similar-case retrieval
- Every training case is vectorized with the same TF-IDF vectorizer and stored in `case_index.joblib`.
- A new case is compared against all of them with cosine similarity, and the top matches are returned with their outcomes.
- Within each retrieved case, sentence-level TF-IDF similarity picks the sentences that best match the input, and the shared content words are highlighted.

### 6. Key-fact extraction
Regular expressions and keyword lists for Hindi legal text (including Devanagari digits and UP-style abbreviations such as `भा0द0वि0`). It extracts IPC / NDPS / POCSO / Arms Act sections, offence category, crime number, police station, custody status and date, chargesheet status, prior-record statements, false-implication claims, co-accused bail, and whether a death or minor is involved.

### 7. Evaluation
The model is evaluated on unseen test data using accuracy, precision, recall, F1 score and a confusion matrix.

---

## Usage

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
python baseline_tfidf.py --data train.csv --test_data test.csv --text_col text --label_col label
```
Use `--model logreg` for Logistic Regression instead of the SVM.

### Build the case index (once, after training)
```bash
python build_case_index.py --data train.csv --text_col text --label_col label
```
Use `--max_cases 50000` to limit index size if memory is tight.

### Predict
```bash
# a row of a CSV file
python predict.py --csv test.csv --row 5 --text_col text

# a text file or a string
python predict.py --file case.txt
python predict.py --text "अभियुक्त ... धारा 302 भा0द0वि0 ..."
```

---

## Important Notes

- **Build the index from training data only.** If test cases are in the index, retrieval would leak the answers the model is evaluated on.
- **Fact extraction is rule-based.** It is transparent and needs no training data, but it can miss unusual wording. Patterns in `legal_facts.py` can be extended as new phrasings are found.
- **Confidence is only as reliable as the model.** Calibration makes the number meaningful, but low-confidence cases should always be reviewed by a person.
- The system supports human decision-making. It does not replace it, and it is not legal advice.

---

## Reference Paper

**HLDC: Hindi Legal Documents Corpus**

Paper: <https://arxiv.org/abs/2204.00806>

Repository: <https://github.com/Exploration-Lab/HLDC>

The original work explored Doc2Vec, SVM, XGBoost, IndicBERT and multi-task learning. This project builds on a TF-IDF + Linear SVM baseline and adds calibrated confidence, similar-case retrieval with sentence-level evidence, and rule-based key-fact extraction.

We also chose this direction because AI-based support for the Indian judicial system is much less explored than in other domains.