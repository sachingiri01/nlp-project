"""
bail_engine.py
Ties together the three outputs:
  1. Prediction + confidence
  2. Similar historical cases with highlighted matching sentences
  3. Extracted key facts
No LLM anywhere: TF-IDF model, cosine similarity, regex rules.
"""
import re

import joblib
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from baseline_tfidf import extract_facts
from legal_facts import extract_key_facts, facts_as_rows
from preprocess import clean_text

LOW_CONF_THRESHOLD = 0.60
_POSITIVE = {"1", "granted", "grant", "true", "yes"}

# very small Hindi stop-list, only used so the highlighting skips filler words
_STOP = set("""
के में की का है हैं को से पर और एवं ने कि यह वह था थे थी तथा भी नहीं ही एक द्वारा लिए
जो तो अपने इस उस कर किया गया गये हो होने कोई तक वे हम आप जब तब कुछ सभी साथ बाद
""".split())


def label_to_name(label) -> str:
    """Map whatever the dataset label is (1/0, 'granted'/'denied', ...) to GRANTED / DENIED."""
    return "GRANTED" if str(label).strip().lower() in _POSITIVE else "DENIED"


# ------------------------------------------------------------ text helpers
def get_raw_facts(text: str) -> str:
    """Accept either a raw dataset cell ('facts-and-arguments': array([...])) or plain Hindi text."""
    if "facts-and-arguments" in text:
        return extract_facts(text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str, chunk_words: int = 25):
    parts = [p.strip() for p in re.split(r"(?<=[।?!])\s+|\n+", text) if len(p.strip()) >= 20]
    if len(parts) >= 3:
        return parts
    words = text.split()   # fallback when the text has no sentence delimiters
    return [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]


def _norm_tok(tok: str) -> str:
    return tok.strip(" ,.;:()[]{}'\"।!?-–—/")


def _content_tokens(sentence: str):
    toks = {_norm_tok(t) for t in sentence.split()}
    return {t for t in toks if len(t) >= 3 and t not in _STOP}


def render_highlight(sentence: str, shared: set) -> str:
    """Wrap shared words in **bold** markers so they stand out in the terminal."""
    out = []
    for part in re.split(r"(\s+)", sentence):
        if part.strip() and _norm_tok(part) in shared:
            part = f"**{part}**"
        out.append(part)
    return "".join(out)


# ------------------------------------------------------------ engine
class BailEngine:
    def __init__(self, model_path="model_tfidf_facts_only.joblib",
                 index_path="case_index.joblib", top_k=3):
        self.pipe = joblib.load(model_path)
        self.vec = self.pipe.named_steps["tfidf"]
        self.top_k = top_k
        try:
            self.index = joblib.load(index_path)
        except FileNotFoundError:
            self.index = None   # prediction + facts still work without the index

    # ---- 1. prediction ---------------------------------------------
    def predict(self, clean: str) -> dict:
        classes = list(self.pipe.classes_)
        calibrated = hasattr(self.pipe, "predict_proba")
        if calibrated:
            proba = self.pipe.predict_proba([clean])[0]
        else:   # LinearSVC has no probabilities -> squash the margin (approximate)
            score = float(self.pipe.decision_function([clean])[0])
            p_pos = 1.0 / (1.0 + np.exp(-score))
            proba = np.array([1 - p_pos, p_pos])
        i = int(np.argmax(proba))
        conf = float(proba[i])
        return {
            "label": label_to_name(classes[i]),
            "confidence": conf,
            "low_confidence": conf < LOW_CONF_THRESHOLD,
            "calibrated": calibrated,
        }

    # ---- 2. similar cases + matching sentences ----------------------
    def _matching_sentences(self, q_sents, c_sents, max_pairs=2, min_sim=0.15):
        if not q_sents or not c_sents:
            return []
        Q = self.vec.transform([clean_text(s, drop_stopwords=False) for s in q_sents])
        C = self.vec.transform([clean_text(s, drop_stopwords=False) for s in c_sents])
        S = linear_kernel(Q, C)
        pairs, used_q, used_c = [], set(), set()
        for flat in np.argsort(-S, axis=None):
            qi, ci = divmod(int(flat), S.shape[1])
            if S[qi, ci] < min_sim or len(pairs) >= max_pairs:
                break
            if qi in used_q or ci in used_c:
                continue
            shared = _content_tokens(q_sents[qi]) & _content_tokens(c_sents[ci])
            used_q.add(qi); used_c.add(ci)
            pairs.append({"your_sentence": q_sents[qi], "case_sentence": c_sents[ci],
                          "score": float(S[qi, ci]), "shared_terms": sorted(shared)})
        return pairs

    def similar_cases(self, raw: str, clean: str, exclude_exact=True) -> list:
        if self.index is None:
            return []
        q = self.vec.transform([clean])
        sims = linear_kernel(q, self.index["matrix"]).ravel()
        k = min(self.top_k + 1, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        q_sents = split_sentences(raw)
        results = []
        for i in top:
            if exclude_exact and sims[i] > 0.999:      # the query itself is in the index
                continue
            c_raw = self.index["raw"][i]
            results.append({
                "case_id": self.index["ids"][i],
                "similarity": float(sims[i]),
                "outcome": self.index["outcomes"][i],
                "matches": self._matching_sentences(q_sents, split_sentences(c_raw)),
            })
            if len(results) == self.top_k:
                break
        return results

    # ---- everything -------------------------------------------------
    def analyze(self, text: str) -> dict:
        raw = get_raw_facts(text)
        clean = clean_text(raw, drop_stopwords=False)
        return {
            "prediction": self.predict(clean),
            "similar_cases": self.similar_cases(raw, clean),
            "facts": extract_key_facts(raw),
        }


# ------------------------------------------------------------ text report
def format_report(res: dict) -> str:
    p = res["prediction"]
    icon = "🟢" if p["label"] == "GRANTED" else "🔴"
    L = ["Predicted Bail Outcome:", "", f"{icon} {p['label']}", "",
         f"Confidence: {p['confidence'] * 100:.0f}%"]
    if p["low_confidence"]:
        L.append("⚠ Low confidence - manual review recommended")
    if not p["calibrated"]:
        L.append("(confidence is an approximation from the SVM margin)")

    L += ["", "Relevant Historical Cases", ""]
    if not res["similar_cases"]:
        L.append("(no case index loaded - run build_case_index.py)")
    for n, c in enumerate(res["similar_cases"], 1):
        L += [f"{n}. Case #{c['case_id']}",
              f"   Similarity: {c['similarity'] * 100:.0f}%",
              f"   Outcome: {c['outcome']}"]
        for m in c["matches"]:
            shared = set(m["shared_terms"])
            L.append(f"   • Your case : {render_highlight(m['your_sentence'][:300], shared)}")
            L.append(f"     Past case : {render_highlight(m['case_sentence'][:300], shared)}")
        L.append("")

    L += ["Extracted Key Facts", ""]
    L += [f"  {k:<28}: {v}" for k, v in facts_as_rows(res["facts"])]
    L += ["", "Decision-support tool only. Not legal advice."]
    return "\n".join(L)