"""Write provenance.json and generation_metadata.json from generated_descriptions.json.

Reads the existing generated_descriptions.json and produces:
  - src/label_descriptions/provenance.json   (array of provenance records)
  - src/label_descriptions/generation_metadata.json  (model/prompt/temperature metadata)

Usage (from project root):
    python -m scripts.write_provenance_metadata
    python -m scripts.write_provenance_metadata --descriptions-path PATH --output-dir DIR
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrors generate_label_descriptions.py)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = (
    "Define the following text classification label in 15-20 words, "
    "focusing only on its semantic core without using the label name itself. "
    "Dataset: {dataset_name}. Label: {label_name}."
)

DATASET_SOURCE_MAP = {
    "ag_news": "wikipedia",
    "yahoo_answers_topics": "wikipedia",
    "SetFit/20_newsgroups": "wikipedia",
    "dbpedia_14": "dbpedia_ontology",
    "banking77": "dataset_documentation",
    "imdb": "psychology_dictionary",
    "sst2": "psychology_dictionary",
    "zeroshot/twitter-financial-news-sentiment": "psychology_dictionary",
    "go_emotions": "ekman_theory",
}

DEFAULT_DESCRIPTIONS_PATH = Path("src/label_descriptions/generated_descriptions.json")
DEFAULT_OUTPUT_DIR = Path("src/label_descriptions")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _is_dry_run_value(value: str) -> bool:
    """Return True if the description value looks like dry-run placeholder data."""
    return isinstance(value, str) and value.startswith("[DRY-RUN]")


def build_provenance_records(descriptions: dict, now_iso: str) -> list:
    """Build a list of provenance records from the descriptions dict.

    Args:
        descriptions: Parsed content of generated_descriptions.json.
            Shape: {dataset: {label_id: {set_a: str, set_b: str}}}
        now_iso: ISO 8601 UTC timestamp to use when no timestamp is available.

    Returns:
        List of provenance record dicts, each with 6 required fields.
    """
    model_a = os.getenv("DESCRIPTION_MODEL_SET_A", "openai/gpt-4o")
    model_b = os.getenv("DESCRIPTION_MODEL_SET_B", "anthropic/claude-3-5-sonnet")

    records = []
    skipped = 0

    for dataset, labels in descriptions.items():
        source_type = DATASET_SOURCE_MAP.get(dataset, "llm_fallback")

        for label_id_str, sets in labels.items():
            label_id = int(label_id_str)

            for set_key, description in sets.items():
                if _is_dry_run_value(description):
                    log.warning(
                        "Skipping dry-run entry: dataset=%s label_id=%s set=%s",
                        dataset, label_id_str, set_key,
                    )
                    skipped += 1
                    continue

                if set_key == "set_a":
                    source_url_or_reference = f"model:{model_a}"
                elif set_key == "set_b":
                    source_url_or_reference = f"model:{model_b}"
                else:
                    source_url_or_reference = f"set:{set_key}"

                records.append({
                    "dataset": dataset,
                    "label_id": label_id,
                    "label_mode": "L2",
                    "source_type": source_type,
                    "source_url_or_reference": source_url_or_reference,
                    "generated_at": now_iso,
                })

    if skipped:
        log.warning("Skipped %d dry-run entries total.", skipped)

    return records


def build_generation_metadata(now_iso: str) -> dict:
    """Build the generation_metadata dict.

    Args:
        now_iso: ISO 8601 UTC timestamp for the generated_at field.

    Returns:
        Dict with model names, prompt template, temperature, and timestamp.
    """
    model_a = os.getenv("DESCRIPTION_MODEL_SET_A", "openai/gpt-4o")
    model_b = os.getenv("DESCRIPTION_MODEL_SET_B", "anthropic/claude-3-5-sonnet")

    return {
        "model_set_a": model_a,
        "model_set_b": model_b,
        "prompt_template": PROMPT_TEMPLATE,
        "temperature": 0,
        "generated_at": now_iso,
    }


def write_provenance_metadata(
    descriptions_path: Path = DEFAULT_DESCRIPTIONS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Main entry point: read descriptions, write provenance + metadata files."""
    descriptions_path = Path(descriptions_path)
    output_dir = Path(output_dir)

    log.info("Reading descriptions from: %s", descriptions_path)
    with open(descriptions_path, encoding="utf-8") as fh:
        descriptions = json.load(fh)

    log.info(
        "Loaded %d dataset(s) from %s",
        len(descriptions),
        descriptions_path,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- provenance.json ---
    records = build_provenance_records(descriptions, now_iso)
    provenance_path = output_dir / "provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    log.info("Provenance saved → %s (%d records)", provenance_path, len(records))

    # --- generation_metadata.json ---
    metadata = build_generation_metadata(now_iso)
    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    log.info("Generation metadata saved → %s", metadata_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Write provenance.json and generation_metadata.json from generated_descriptions.json."
    )
    p.add_argument(
        "--descriptions-path",
        default=str(DEFAULT_DESCRIPTIONS_PATH),
        help="Path to generated_descriptions.json (default: %(default)s).",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write output files (default: %(default)s).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    write_provenance_metadata(
        descriptions_path=Path(args.descriptions_path),
        output_dir=Path(args.output_dir),
    )
