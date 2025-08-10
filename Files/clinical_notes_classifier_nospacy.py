#!/usr/bin/env python3
"""
Clinical Notes → Multi-label Phenotype Classifier (No tensors, No spaCy)
-----------------------------------------------------------------------
- Portfolio-ready NLP using scikit-learn only (Logistic Regression, TF-IDF)
- Handles section extraction and simple negation (pure-Python)
- Works out-of-the-box with synthetic demo data; or pass your own CSV

Expected CSV format (if using your own data):
    note_id,str  | note_text,str | diabetes,int(0/1) | chf,int | copd,int

Usage examples:
    # 1) Run with demo data (default)
    python clinical_notes_classifier_nospacy.py

    # 2) Run with your CSV and save the model
    python clinical_notes_classifier_nospacy.py --csv notes.csv --text-col note_text --labels diabetes chf copd --save-model notes_phenotype_clf.joblib

    # 3) Just train/eval without saving
    python clinical_notes_classifier_nospacy.py --no-save

Dependencies:
    pip install scikit-learn pandas numpy joblib
"""

import re
import os
import json
import joblib
import string
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import argparse

warnings.filterwarnings("ignore")

# -----------------------------
# 1) Synthetic demo dataset
# -----------------------------

def demo_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Create HIPAA-safe synthetic progress notes + labels.
    Labels: diabetes, chf, copd (multi-label)
    """
    rng = np.random.default_rng(seed)

    templates = [
        ("HPI: Patient with {cond}. Denies {neg}. "
         "Assessment: {assess}. Plan: {plan}."),
        ("Subjective: {cond}. ROS negative for {neg}. "
         "Assessment/Plan: {assess}; {plan}."),
        ("Assessment: {assess}. PMH notable for {cond}. "
         "Plan: {plan}. Dispo: home."),
    ]

    cond_pool = [
        ("type 2 diabetes", "hypoglycemia episodes", "optimize metformin"),
        ("congestive heart failure", "chest pain", "increase furosemide"),
        ("COPD with chronic bronchitis", "fever or chills", "tiotropium daily"),
        ("no significant PMH", "dyspnea", "routine follow-up"),
    ]

    rows = []
    for i in range(n):
        t = templates[i % len(templates)]
        c = cond_pool[rng.integers(0, len(cond_pool))]
        text = t.format(cond=c[0], neg=c[1], assess="stable", plan=c[2])

        # Add some section headers and noise
        if rng.random() < 0.4:
            text = f"MEDS: lisinopril, metformin.\n{text}"
        if rng.random() < 0.4:
            text += "\nAllergies: NKDA."

        # Labels (weakly tied to chosen condition)
        y_diab = int("diabetes" in c[0] or ("metformin" in text and rng.random() < 0.5))
        y_chf = int("heart failure" in c[0] or ("furosemide" in text and rng.random() < 0.5))
        y_copd = int("COPD" in c[0] or ("tiotropium" in text and rng.random() < 0.5))

        # Negation flips a fraction of positives
        if "Denies" in text or "negative for" in text:
            if rng.random() < 0.35:
                y_diab = max(0, y_diab - 1)
            if rng.random() < 0.35:
                y_chf = max(0, y_chf - 1)
            if rng.random() < 0.35:
                y_copd = max(0, y_copd - 1)

        rows.append({"note_id": f"N{i:04d}", "note_text": text,
                     "diabetes": y_diab, "chf": y_chf, "copd": y_copd})

    return pd.DataFrame(rows)


# -----------------------------
# 2) Text preprocessing
# -----------------------------

SECTION_HEADERS = [
    "HPI", "ASSESSMENT", "PLAN", "ASSESSMENT/PLAN", "A/P",
    "SUBJECTIVE", "ROS", "PMH", "MEDS", "ALLERGIES", "DISPO"
]
SECTION_PATTERN = re.compile(
    r"(?P<header>^|\n)(?P<name>(" + "|".join([re.escape(h) for h in SECTION_HEADERS]) +
    r"))\s*:?", flags=re.IGNORECASE
)

class SectionExtractor(BaseEstimator, TransformerMixin):
    """Extracts and reorders salient sections (Assessment + Plan weighted)."""
    def __init__(self, keep=("HPI","ASSESSMENT","PLAN","ASSESSMENT/PLAN","SUBJECTIVE"), weight_plan=2):
        self.keep = set(s.upper() for s in keep)
        self.weight_plan = weight_plan

    def _split_sections(self, text: str) -> List[Tuple[str, str]]:
        sections = []
        matches = list(SECTION_PATTERN.finditer(text))
        for i, m in enumerate(matches):
            name = m.group("name").upper()
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((name, body))
        if not matches:
            sections = [("FREE_TEXT", text)]
        return sections

    def transform(self, X, y=None):
        out = []
        for txt in X:
            secs = self._split_sections(txt)
            selected = [b for (h,b) in secs if h in self.keep or h == "FREE_TEXT"]
            # Weight plan/assessment text
            boosted = []
            for b in selected:
                boosted.append(b)
                if re.search(r"\b(plan|assessment)\b", b, re.I):
                    boosted.append((" " + b) * (self.weight_plan-1))
            out.append("\n".join(boosted))
        return out

    def fit(self, X, y=None):
        return self


class SimpleNegationTagger(BaseEstimator, TransformerMixin):
    """
    Pure-Python negation tagger (no spaCy).
    Tokenizes with a simple regex; tags a window after negation cues.
    """
    def __init__(self, window: int = 6):
        self.window = window
        self.token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        self.neg_bigrams = {"negative for", "rule out"}
        self.neg_unigrams = {"no", "denies", "without", "not"}

    def _find_negation_positions(self, lowered_tokens: List[str]) -> List[int]:
        positions = []
        for i, tok in enumerate(lowered_tokens):
            if tok in self.neg_unigrams:
                positions.append(i)
            if i + 1 < len(lowered_tokens):
                pair = f"{tok} {lowered_tokens[i+1]}"
                if pair in self.neg_bigrams:
                    positions.append(i)
        return positions

    def _tag(self, text: str) -> str:
        tokens = self.token_re.findall(text)
        lowered = [t.lower() for t in tokens]
        neg_positions = self._find_negation_positions(lowered)

        tagged = tokens[:]
        for i in neg_positions:
            end = min(i + 1 + self.window, len(tokens))
            for j in range(i + 1, end):
                if tokens[j] in string.punctuation:
                    break
                tagged[j] = f"{tokens[j]}_NEG"
        return " ".join(tagged)

    def transform(self, X, y=None):
        return [self._tag(x) for x in X]

    def fit(self, X, y=None):
        return self


# -----------------------------
# 3) Build pipeline
# -----------------------------

def build_pipeline() -> Pipeline:
    """
    Pipeline:
        SectionExtractor -> SimpleNegationTagger -> Tfidf -> OneVsRest(LogReg)
    """
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=300, solver="liblinear", class_weight="balanced")
    )
    pipe = Pipeline(steps=[
        ("sectioner", SectionExtractor()),
        ("negation", SimpleNegationTagger(window=6)),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_df=0.9,
            strip_accents="unicode",
            sublinear_tf=True,
        )),
        ("clf", clf)
    ])
    return pipe


# -----------------------------
# 4) Train / Evaluate
# -----------------------------

@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42

def run_experiment(df: pd.DataFrame, cfg: TrainConfig, label_cols: List[str]):
    assert all(col in df.columns for col in label_cols), f"Missing label columns in data: {label_cols}"
    assert "note_text" in df.columns, "Data must contain a 'note_text' column."

    X = df["note_text"].astype(str).tolist()
    Y = df[label_cols].astype(int).values
    labels = label_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)
    pred = (prob >= 0.5).astype(int)

    # Metrics
    micro_f1 = f1_score(y_test, pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)

    roc_aucs = {}
    for i, lab in enumerate(labels):
        try:
            roc_aucs[lab] = roc_auc_score(y_test[:, i], prob[:, i])
        except ValueError:
            roc_aucs[lab] = float("nan")

    print("\n=== Metrics ===")
    print(f"Micro-F1: {micro_f1:.3f} | Macro-F1: {macro_f1:.3f}")
    print("ROC-AUC per class:", {k: (None if np.isnan(v) else round(v,3)) for k,v in roc_aucs.items()})

    print("\n=== Per-class report (threshold=0.5) ===")
    print(classification_report(y_test, pred, target_names=labels, zero_division=0))

    return pipe, labels


# -----------------------------
# 5) Model explainability (top n-grams)
# -----------------------------

def top_features(pipe: Pipeline, labels: List[str], k: int = 12):
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: OneVsRestClassifier = pipe.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())

    print("\n=== Top n-grams per class (positive/negative) ===")
    for i, lab in enumerate(labels):
        lr: LogisticRegression = clf.estimators_[i]
        coefs = lr.coef_.ravel()
        top_pos = np.argsort(coefs)[-k:][::-1]
        top_neg = np.argsort(coefs)[:k]
        print(f"\n[{lab}] + predictors:")
        for idx in top_pos:
            print(f"  {feature_names[idx]:<30} {coefs[idx]:.3f}")
        print(f"[{lab}] − predictors:")
        for idx in top_neg:
            print(f"  {feature_names[idx]:<30} {coefs[idx]:.3f}")


# -----------------------------
# 6) Inference utility
# -----------------------------

def predict_note(pipe: Pipeline, text: str, labels: List[str], threshold: float = 0.5):
    prob = pipe.predict_proba([text])[0]
    pred = (prob >= threshold).astype(int)
    return dict(zip(labels, [float(p) for p in prob])), dict(zip(labels, [int(x) for x in pred]))


# -----------------------------
# 7) CLI + Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Clinical Notes Phenotype Classifier (no tensors, no spaCy)")
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV with note_text and label columns")
    ap.add_argument("--text-col", type=str, default="note_text", help="Text column name (default: note_text)")
    ap.add_argument("--labels", nargs="+", default=["diabetes", "chf", "copd"], help="Label column names")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default: 0.2)")
    ap.add_argument("--random-state", type=int, default=7, help="Random state (default: 7)")
    ap.add_argument("--k-top", type=int, default=12, help="Top n-grams to display per class (default: 12)")
    ap.add_argument("--save-model", type=str, default="notes_phenotype_clf.joblib", help="Path to save model (joblib)")
    ap.add_argument("--no-save", action="store_true", help="Do not save the trained model")
    return ap.parse_args()

def load_data_from_csv(path: str, text_col: str, label_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [text_col] + label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df = df.rename(columns={text_col: "note_text"})
    for c in label_cols:
        df[c] = df[c].astype(int)
    df["note_text"] = df["note_text"].astype(str)
    return df

def main():
    args = parse_args()

    if args.csv:
        print(f"Loading data from {args.csv} ...")
        df = load_data_from_csv(args.csv, args.text_col, args.labels)
    else:
        print("No CSV provided; generating synthetic demo data ...")
        df = demo_data(n=500, seed=42)

    # Train/evaluate
    pipe, labels = run_experiment(df, TrainConfig(test_size=args.test_size, random_state=args.random_state), args.labels)

    # Explainability
    top_features(pipe, labels, k=args.k_top)

    # Try a few demo notes
    examples = [
        "HPI: Patient with type 2 diabetes. Denies hypoglycemia. Assessment: stable. Plan: optimize metformin.",
        "Subjective: Progressive dyspnea and orthopnea. Assessment/Plan: increase furosemide. Dispo: home.",
        "Assessment: COPD exacerbation likely. Plan: tiotropium daily. ROS negative for fever.",
        "HPI: No history of diabetes or heart failure. Plan: routine follow-up."
    ]
    for t in examples:
        proba, pred = predict_note(pipe, t, labels, threshold=0.5)
        print("\nNOTE:", t)
        print("Prob:", json.dumps(proba, indent=2))
        print("Pred:", pred)

    # Persist
    if not args.no_save:
        out_path = args.save_model
        joblib.dump({"pipeline": pipe, "labels": labels}, out_path)
        print(f"\nSaved model → {out_path}")

if __name__ == "__main__":
    main()
