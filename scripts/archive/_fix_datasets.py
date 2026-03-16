import re, pathlib

# ── analyze_error_patterns.py ────────────────────────────────────────────────
p = pathlib.Path("scripts/analyze_error_patterns.py")
txt = p.read_text(encoding="utf-8")

# DATASET_NAMES
txt = txt.replace(
    '"yahoo_answers_topics": "Yahoo Answers"\n}',
    '"yahoo_answers_topics": "Yahoo Answers",\n    "imdb": "IMDB",\n    "sst2": "SST-2",\n}'
)

# label_key_map in get_label_names
txt = txt.replace(
    '"yahoo_answers_topics": "yahoo_answers_topics"\n    }',
    '"yahoo_answers_topics": "yahoo_answers_topics",\n        "imdb": "imdb",\n        "sst2": "sst2",\n    }'
)

p.write_text(txt, encoding="utf-8")
print("analyze_error_patterns.py updated")

# ── generate_confusion_matrices.py ───────────────────────────────────────────
p2 = pathlib.Path("scripts/generate_confusion_matrices.py")
txt2 = p2.read_text(encoding="utf-8")

# DATASET_NAMES
txt2 = txt2.replace(
    '"go_emotions": "GoEmotions"\n}',
    '"go_emotions": "GoEmotions",\n    "imdb": "IMDB",\n    "sst2": "SST-2",\n}'
)

# label_key_map in get_label_names
txt2 = txt2.replace(
    '"go_emotions": "go_emotions"\n    }',
    '"go_emotions": "go_emotions",\n        "imdb": "imdb",\n        "sst2": "sst2",\n    }'
)

p2.write_text(txt2, encoding="utf-8")
print("generate_confusion_matrices.py updated")

# ── CD diagram caption ────────────────────────────────────────────────────────
p3 = pathlib.Path("scripts/generate_critical_difference_diagram.py")
txt3 = p3.read_text(encoding="utf-8")
txt3 = txt3.replace("across eight text classification datasets", "across nine text classification datasets")
txt3 = txt3.replace("across seven text classification datasets", "across nine text classification datasets")
p3.write_text(txt3, encoding="utf-8")
print("generate_critical_difference_diagram.py caption updated")
