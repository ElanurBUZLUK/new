import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd
import spacy
from wordfreq import zipf_frequency

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw_word_lists"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

SPACY_MODELS = {
    "English": "en_core_web_trf",
    "German": "de_dep_news_trf",
    "Turkish": "tr_core_news_lg",
    # "Spanish": "es_dep_news_trf",
}

def iter_raw_words(txt_path: Path):
    with txt_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                w = cell.strip()
                if w:
                    yield w

def normalize_word(w: str) -> str:
    w = w.strip().lower()
    w = re.sub(r"^[^\w]+|[^\w]+$", "", w, flags=re.UNICODE)
    w = "".join(ch for ch in w if ch.isalpha() or ch in {"'", "-"})
    return w

def label_difficulty(zipf: float) -> str:
    if zipf <= 2:
        return "advanced"
    if zipf < 4:
        return "intermediate"
    return "beginner"

def create_clean_word_list(language: str, max_words: int | None = None):
    if language not in SPACY_MODELS:
        raise ValueError(f"Unknown language: {language}. Add to SPACY_MODELS.")

    model_name = SPACY_MODELS[language]
    lang_code = model_name.split("_", 1)[0]

    raw_file = RAW_DIR / language / f"{language}.txt"
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw file: {raw_file}")

    words = []
    for i, w in enumerate(iter_raw_words(raw_file)):
        nw = normalize_word(w)
        if nw:
            words.append(nw)
        if max_words and i >= max_words:
            break

    df = pd.DataFrame({"word": words}).drop_duplicates()

    nlp = spacy.load(model_name, disable=["parser", "ner", "textcat"])
    lemmas = []
    for doc in nlp.pipe(df["word"].tolist(), batch_size=2048):
        if len(doc) == 0:
            lemmas.append("")
        else:
            lemmas.append(doc[0].lemma_.lower())

    df["lemma"] = lemmas
    df = df[df["lemma"] != ""]

    df["zipf"] = df["lemma"].map(lambda x: float(zipf_frequency(x, lang_code)))

    df = df[df["zipf"] > 0]
    df = df.sort_values("zipf", ascending=False).drop_duplicates(subset=["lemma"], keep="first")

    df["word_difficulty"] = df["zipf"].map(label_difficulty)

    out = df[["lemma", "word_difficulty"]].rename(columns={"lemma": "word"})

    out_dir = CLEAN_DIR / language
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "word_list_clean.json"
    out.to_json(out_path, orient="records", force_ascii=False, indent=2)

    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--max_words", type=int, default=None, help="debug için kısıtla")
    args = ap.parse_args()

    for lang in args.languages:
        p = create_clean_word_list(lang, max_words=args.max_words)
        print(f"[OK] {lang}: {p}")

if __name__ == "__main__":
    main()
