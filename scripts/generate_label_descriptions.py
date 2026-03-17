"""Generate LLM-based label descriptions for zero-shot classification research.

Required packages (install separately):
    openai, anthropic, python-dotenv

Usage (run from project root):
    python -m scripts.generate_label_descriptions [--dataset DATASET] [--set {a,b,both}]
                                                   [--dry-run] [--output PATH]
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# NOTE: Legacy PROMPT_L2 / PROMPT_L3 constants have been removed.
# All prompts are loaded from src/label_descriptions/prompt_templates.yaml
# via TaskAwareLabelGenerator. Those YAML templates are the single source of truth.

OUTPUT_DIR = Path("src/label_descriptions")

# Authoritative source mapping (Requirement 8.4)
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


# ---------------------------------------------------------------------------
# DescriptionGenerator
# ---------------------------------------------------------------------------
class DescriptionGenerator:
    """Generates label descriptions via LLM APIs.

    Supports:
    - OpenRouter (primary): model names prefixed with ``openai/`` or ``anthropic/``
    - Direct OpenAI (fallback): model names starting with ``gpt-``
    - Direct Anthropic (fallback): model names starting with ``claude-``
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        self.model = model
        self._client = None
        self._backend = self._detect_backend(model)

        if base_url:
            # Explicit override
            self._base_url = base_url
            self._api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            self._backend = "openai_compat"
        elif self._backend == "openrouter":
            self._base_url = self.OPENROUTER_BASE_URL
            self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        elif self._backend == "openai":
            self._base_url = None  # use default
            self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        elif self._backend == "anthropic":
            self._base_url = None
            self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Cannot determine backend for model: {model!r}")

        log.debug("DescriptionGenerator: model=%s backend=%s", model, self._backend)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend(model: str) -> str:
        """Infer backend from model name prefix."""
        if model.startswith("openai/") or model.startswith("anthropic/"):
            return "openrouter"
        if model.startswith("gpt-"):
            return "openai"
        if model.startswith("claude-"):
            return "anthropic"
        # Default to openrouter for unknown prefixes (e.g. meta-llama/)
        return "openrouter"

    def _get_openai_client(self):
        """Lazy-init OpenAI-compatible client."""
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415

            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _get_anthropic_client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            import anthropic  # noqa: PLC0415

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _call_openai_compat(self, prompt: str) -> str:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str) -> str:
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _call_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with exponential backoff on rate-limit errors."""
        for attempt in range(max_retries):
            try:
                if self._backend == "anthropic":
                    return self._call_anthropic(prompt)
                else:
                    return self._call_openai_compat(prompt)
            except Exception as exc:
                err_str = str(exc).lower()
                is_rate_limit = "rate" in err_str or "429" in err_str or "too many" in err_str
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    log.warning("Rate limit hit, retrying in %ds (attempt %d/%d)…", wait, attempt + 1, max_retries)
                    time.sleep(wait)
                else:
                    raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, dataset_name: str, label_name: str, prompt_template: str) -> str:
        """Generate a description for a single label.

        Args:
            dataset_name: Dataset identifier (e.g. ``"ag_news"``).
            label_name: Human-readable label name (e.g. ``"world"``).
            prompt_template: Prompt template to use.

        Returns:
            Generated description string.
        """
        prompt = prompt_template.format(dataset_name=dataset_name, label_name=label_name)
        return self._call_with_retry(prompt)

    @staticmethod
    def _parse_l3_response(response: str) -> list:
        """Parse L3 response into a list of sentences.
        
        Handles two formats:
        1. Newline-separated (one sentence per line, optionally numbered)
        2. Single paragraph (sentences separated by '. ')
        """
        import re
        text = response.strip()
        
        # Try newline-based split first (numbered or plain lines)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        cleaned = []
        for line in lines:
            # Strip numbering like "1.", "2.", "3." or "1)", "2)", "3)"
            stripped = re.sub(r'^\d+[.)]\s*', '', line).strip()
            if stripped and len(stripped.split()) >= 5:
                cleaned.append(stripped)
        
        if len(cleaned) >= 3:
            return cleaned[:3]
        
        # Fallback: split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 5]
        
        if len(sentences) >= 3:
            return sentences[:3]
        
        # Last resort: return whatever we have
        return cleaned if cleaned else sentences

    def generate_multi(self, dataset_name: str, label_name: str, prompt_template: str) -> list:
        """Generate 3 different descriptions for a single label (L3).

        Args:
            dataset_name: Dataset identifier.
            label_name: Human-readable label name.
            prompt_template: Prompt template string (from prompt_templates.yaml via TaskAwareLabelGenerator).

        Returns:
            List of 3 description strings.
        """
        prompt = prompt_template.format(dataset_name=dataset_name, label_name=label_name)
        response = self._call_with_retry(prompt)
        descriptions = self._parse_l3_response(response)
        
        if len(descriptions) < 3:
            log.warning("  L3 returned fewer than 3 descriptions (%d), retrying once...", len(descriptions))
            retry_response = self._call_with_retry(prompt)
            descriptions = self._parse_l3_response(retry_response)
            if len(descriptions) < 3:
                raise ValueError(
                    f"L3 generation returned fewer than 3 descriptions after retry "
                    f"(got {len(descriptions)}). Raw response: {retry_response!r}"
                )
        
        return descriptions[:3]



# ---------------------------------------------------------------------------
# AuthoritativeSourceMapper
# ---------------------------------------------------------------------------
class AuthoritativeSourceMapper:
    """Maps datasets to their authoritative description sources."""

    DATASET_SOURCE_MAP = DATASET_SOURCE_MAP

    def get_source_type(self, dataset_name: str) -> str:
        return self.DATASET_SOURCE_MAP.get(dataset_name, "llm_fallback")

    def fetch_authoritative_description(
        self, dataset_name: str, label_name: str
    ) -> tuple:
        """Attempt to fetch an authoritative description.

        For now returns ``(None, source_type)`` — actual fetching is complex
        and the LLM fallback handles generation.  The source_type is recorded
        in provenance.

        Returns:
            ``(description_or_None, source_type)``
        """
        source_type = self.get_source_type(dataset_name)
        # Actual fetching not implemented; fall back to LLM generation.
        return (None, source_type)


# ---------------------------------------------------------------------------
# ProvenanceRecorder
# ---------------------------------------------------------------------------
class ProvenanceRecorder:
    """Records provenance for each generated description."""

    def __init__(self):
        self._records: list = []

    def record(
        self,
        dataset: str,
        label_id: int,
        label_mode: str,
        source_type: str,
        source_url_or_reference: str,
        generated_at: str,
    ) -> None:
        self._records.append(
            {
                "dataset": dataset,
                "label_id": label_id,
                "label_mode": label_mode,
                "source_type": source_type,
                "source_url_or_reference": source_url_or_reference,
                "generated_at": generated_at,
            }
        )

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self._records, fh, indent=2, ensure_ascii=False)
        log.info("Provenance saved → %s (%d records)", out, len(self._records))


# ---------------------------------------------------------------------------
# GenerationMetadata
# ---------------------------------------------------------------------------
class GenerationMetadata:
    """Stores and persists metadata about a generation run."""

    def __init__(self, model: str, prompt_template: str, temperature: float):
        self.model = model
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_template": self.prompt_template,
            "temperature": self.temperature,
            "generated_at": self.generated_at,
        }

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        log.info("Generation metadata saved → %s", out)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def _extract_label_names(label_dict: dict) -> dict:
    """Extract plain label name strings from a ``name_only`` label dict."""
    return {label_id: texts[0] for label_id, texts in label_dict.items()}


def generate_descriptions(
    dataset_filter: str = None,
    level: str = "both",
    dry_run: bool = False,
    output_path: str = None,
) -> dict:
    """Generate L2 and/or L3 descriptions for all (or one) dataset(s).

    Args:
        dataset_filter: If given, only process this dataset name.
        level: ``"l2"``, ``"l3"``, or ``"both"``.
        dry_run: Print prompts without calling the API.
        output_path: Where to write the output JSON.

    Returns:
        ``{dataset_name: {label_id: {"l2": str, "l3": [str, str, str]}}}``
    """
    from src.labels import LABEL_SETS  # noqa: PLC0415

    output_path = output_path or str(OUTPUT_DIR / "generated_descriptions.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use GPT-4o-mini for both L2 and L3
    model = os.getenv("DESCRIPTION_MODEL", "openai/gpt-4o-mini")
    log.info("Model: %s", model)
    log.info("Level: %s", level)

    # Build generator (only if not dry-run)
    generator = DescriptionGenerator(model) if not dry_run else None

    from src.label_descriptions.task_aware_label_generator import (  # noqa: PLC0415
        DatasetTaskTypeConfig,
        TaskAwareLabelGenerator,
        TemplateStore,
    )

    template_path = Path("src/label_descriptions/prompt_templates.yaml")
    dataset_cfg_path = Path("src/label_descriptions/dataset_task_types.yaml")

    template_store = TemplateStore.load(template_path)
    dataset_cfg = DatasetTaskTypeConfig.load(dataset_cfg_path)
    task_aware = (
        TaskAwareLabelGenerator(template_store=template_store, dataset_config=dataset_cfg, llm_generator=generator)
        if not dry_run
        else TaskAwareLabelGenerator(template_store=template_store, dataset_config=dataset_cfg, llm_generator=None)
    )

    source_mapper = AuthoritativeSourceMapper()
    provenance = ProvenanceRecorder()

    meta_l2 = GenerationMetadata(model, "task_aware", temperature=0)
    meta_l3 = GenerationMetadata(model, "task_aware", temperature=0)

    results: dict = {}
    datasets = (
        {dataset_filter: LABEL_SETS[dataset_filter]}
        if dataset_filter
        else LABEL_SETS
    )

    for dataset_name, modes in datasets.items():
        log.info("=== Dataset: %s ===", dataset_name)
        name_only = modes["name_only"]
        label_names = _extract_label_names(name_only)
        results[dataset_name] = {}

        _, source_type = source_mapper.fetch_authoritative_description(dataset_name, "")
        effective_source = "llm_fallback"
        generated_at = datetime.now(timezone.utc).isoformat()

        for label_id in sorted(name_only.keys()):
            label_name = label_names[label_id]
            entry: dict = {}

            if dry_run:
                task_type, prompt_l2, prompt_l3 = task_aware.build_prompts(dataset_id=dataset_name, label_text=label_name)
                log.info("  [DRY-RUN] label %d %r", label_id, label_name)
                if level in ("l2", "both"):
                    entry["l2"] = f"[DRY-RUN] task_type={task_type} prompt={prompt_l2}"
                if level in ("l3", "both"):
                    entry["l3"] = [f"[DRY-RUN] task_type={task_type} prompt={prompt_l3}"]
            else:
                try:
                    l2, l3_list = task_aware.generate_for_label(dataset_id=dataset_name, label_text=label_name)
                except Exception as exc:
                    log.error("  Generation failed for dataset=%s label_id=%s label=%r: %s", dataset_name, label_id, label_name, exc)
                    results[dataset_name][label_id] = {"error": str(exc)}
                    continue

                if level in ("l2", "both"):
                    entry["l2"] = l2
                    provenance.record(
                        dataset=dataset_name,
                        label_id=label_id,
                        label_mode="L2",
                        source_type=effective_source,
                        source_url_or_reference=f"model:{model}",
                        generated_at=generated_at,
                    )

                if level in ("l3", "both"):
                    entry["l3"] = l3_list
                    provenance.record(
                        dataset=dataset_name,
                        label_id=label_id,
                        label_mode="L3",
                        source_type=effective_source,
                        source_url_or_reference=f"model:{model}",
                        generated_at=generated_at,
                    )

            results[dataset_name][label_id] = entry

    # Persist outputs
    if not dry_run:
        provenance.save(str(OUTPUT_DIR / "provenance.json"))

        task_aware.metadata_logger.save_json(OUTPUT_DIR / "generation_operations.json")

        # Build generation_metadata.json
        combined_meta: dict = {}
        if level in ("l2", "both"):
            combined_meta["l2"] = meta_l2.to_dict()
        if level in ("l3", "both"):
            combined_meta["l3"] = meta_l3.to_dict()
        meta_path = OUTPUT_DIR / "generation_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as _fh:
            json.dump(combined_meta, _fh, indent=2, ensure_ascii=False)
        log.info("Generation metadata saved → %s", meta_path)

    # Write main output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    log.info("Descriptions saved → %s", out)

    # Summary
    total_labels = sum(len(v) for v in results.values())
    log.info(
        "Summary: %d dataset(s), %d label(s) processed (level=%s, dry_run=%s)",
        len(results),
        total_labels,
        level,
        dry_run,
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate LLM-based label descriptions for zero-shot classification."
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Generate for a specific dataset only (default: all 9 datasets).",
    )
    p.add_argument(
        "--level",
        choices=["l2", "l3", "both"],
        default="both",
        help="Which description level to generate: l2 (single), l3 (multi-aspect), or both (default: both).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the API.",
    )
    p.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "generated_descriptions.json"),
        help="Output JSON file path.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    generate_descriptions(
        dataset_filter=args.dataset,
        level=args.level,
        dry_run=args.dry_run,
        output_path=args.output,
    )
