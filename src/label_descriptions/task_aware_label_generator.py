from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


SUPPORTED_TASK_TYPES = ("topic", "entity", "sentiment", "emotion", "intent", "newsgroup")


class TemplateConfigError(ValueError):
    pass


class DatasetConfigError(ValueError):
    pass


class GenerationError(RuntimeError):
    pass


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    task_type: str
    l2: str
    l3: str

    def to_dict(self) -> Dict[str, str]:
        return {"l2": self.l2, "l3": self.l3}


@dataclass(frozen=True)
class GenerationRecord:
    generation_id: str
    generated_at: str
    dataset: str
    task_type: str
    label_text: str
    template_l2: str
    template_l3: str
    l2_description: str
    l3_description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "generated_at": self.generated_at,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "label_text": self.label_text,
            "template_l2": self.template_l2,
            "template_l3": self.template_l3,
            "l2_description": self.l2_description,
            "l3_description": self.l3_description,
        }


class TemplateStore:
    def __init__(self, templates: Dict[str, PromptTemplate]):
        self._templates = templates

    @staticmethod
    def load(path: str | Path) -> "TemplateStore":
        p = Path(path)
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TemplateConfigError(f"Failed to load templates from {str(p)!r}: {exc}") from exc

        if not isinstance(raw, dict):
            raise TemplateConfigError("Template configuration must be a mapping")

        supported = raw.get("supported_task_types")
        templates_raw = raw.get("templates")

        if supported is None or templates_raw is None:
            raise TemplateConfigError("Template configuration must contain 'supported_task_types' and 'templates'")

        if not isinstance(supported, list) or any(not isinstance(x, str) for x in supported):
            raise TemplateConfigError("'supported_task_types' must be a list of strings")

        supported_tuple = tuple(supported)

        missing_supported = [t for t in SUPPORTED_TASK_TYPES if t not in supported_tuple]
        if missing_supported:
            raise TemplateConfigError(
                "Template configuration missing required task types in 'supported_task_types': "
                + ", ".join(missing_supported)
            )

        if not isinstance(templates_raw, dict):
            raise TemplateConfigError("'templates' must be a mapping")

        templates: Dict[str, PromptTemplate] = {}
        for task_type in SUPPORTED_TASK_TYPES:
            entry = templates_raw.get(task_type)
            if entry is None:
                raise TemplateConfigError(f"Missing template entry for task type: {task_type!r}")
            if not isinstance(entry, dict):
                raise TemplateConfigError(f"Template entry for {task_type!r} must be a mapping")
            l2 = entry.get("l2")
            l3 = entry.get("l3")
            if not isinstance(l2, str) or not isinstance(l3, str):
                raise TemplateConfigError(f"Template entry for {task_type!r} must include string keys 'l2' and 'l3'")

            templates[task_type] = PromptTemplate(task_type=task_type, l2=l2, l3=l3)

        return TemplateStore(templates=templates)

    def serialize(self) -> Dict[str, Any]:
        return {
            "supported_task_types": list(SUPPORTED_TASK_TYPES),
            "templates": {k: v.to_dict() for k, v in self._templates.items()},
        }

    def dumps_yaml(self) -> str:
        return yaml.safe_dump(self.serialize(), sort_keys=True, allow_unicode=True)

    def get(self, task_type: str) -> PromptTemplate:
        if task_type not in SUPPORTED_TASK_TYPES:
            raise TemplateConfigError(
                f"Unsupported task type: {task_type!r}. Valid task types: {', '.join(SUPPORTED_TASK_TYPES)}"
            )
        return self._templates[task_type]


class DatasetTaskTypeConfig:
    def __init__(self, mapping: Dict[str, str]):
        self._mapping = mapping

    @staticmethod
    def load(path: str | Path) -> "DatasetTaskTypeConfig":
        p = Path(path)
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DatasetConfigError(f"Failed to load dataset task types from {str(p)!r}: {exc}") from exc

        if not isinstance(raw, dict) or "datasets" not in raw:
            raise DatasetConfigError("Dataset task type configuration must be a mapping with top-level key 'datasets'")

        datasets = raw["datasets"]
        if not isinstance(datasets, dict):
            raise DatasetConfigError("'datasets' must be a mapping of dataset_id -> task_type")

        mapping: Dict[str, str] = {}
        for dataset_id, task_type in datasets.items():
            if not isinstance(dataset_id, str) or not isinstance(task_type, str):
                raise DatasetConfigError("Dataset ids and task types must be strings")
            if task_type not in SUPPORTED_TASK_TYPES:
                raise DatasetConfigError(
                    f"Unsupported task type {task_type!r} for dataset {dataset_id!r}. "
                    f"Valid task types: {', '.join(SUPPORTED_TASK_TYPES)}"
                )
            mapping[dataset_id] = task_type

        return DatasetTaskTypeConfig(mapping=mapping)

    def get_task_type(self, dataset_id: str) -> str:
        if dataset_id not in self._mapping:
            raise DatasetConfigError(
                f"Missing task type for dataset {dataset_id!r} in dataset configuration. "
                f"Add it to dataset_task_types.yaml."
            )
        return self._mapping[dataset_id]


BANNED_PATTERNS = (
    "this category",
    "in this context",
    "examples include",
    "this label",
    "this class",
    "this type",
    "refers to",
    "such as",
    "for example",
    "this text",
    "this question",
    "this request may",
)

class ValidationEngine:
    def __init__(
        self,
        generic_prefixes: Optional[Tuple[str, ...]] = None,
        min_words: int = 7,
        max_words: int = 35,
    ):
        self._generic_prefixes = generic_prefixes or ()
        self._min_words = min_words
        self._max_words = max_words

    def validate_single(self, label_text: str, description: str) -> None:
        if not isinstance(description, str) or not description.strip():
            raise ValidationError("Description must be a non-empty string")

        desc = description.strip()

        # 1. Label must appear in description (case-insensitive)
        label_variants = {label_text.lower(), label_text.lower().replace("_", " ")}
        desc_lower = desc.lower()
        if not any(v in desc_lower for v in label_variants):
            raise ValidationError(
                f"Description must contain label text '{label_text}'. Got: {desc[:80]!r}"
            )

        # 2. Word count check
        word_count = len(desc.split())
        if word_count < self._min_words:
            raise ValidationError(
                f"Description too short ({word_count} words, min={self._min_words}): {desc[:80]!r}"
            )
        if word_count > self._max_words:
            raise ValidationError(
                f"Description too long ({word_count} words, max={self._max_words}): {desc[:80]!r}"
            )

        # 3. Banned patterns
        for pattern in BANNED_PATTERNS:
            if pattern in desc_lower:
                raise ValidationError(
                    f"Description contains banned pattern '{pattern}': {desc[:80]!r}"
                )

        # 4. Generic prefix check
        for pfx in self._generic_prefixes:
            if desc_lower.startswith(pfx):
                raise ValidationError(
                    f"Description starts with generic prefix '{pfx}': {desc[:80]!r}"
                )

    def validate_pair(self, label_text: str, l2: str, l3: str) -> None:
        self.validate_single(label_text, l2)
        self.validate_single(label_text, l3)
        if l2.strip() == l3.strip():
            raise ValidationError("L2 and L3 descriptions must be distinct")

    def validate_l3_list(self, label_text: str, l2: str, l3_list: List[str]) -> None:
        self.validate_single(label_text, l2)
        if not isinstance(l3_list, list) or not l3_list:
            raise ValidationError("L3 must be a non-empty list of strings")
        for idx, d in enumerate(l3_list):
            try:
                self.validate_single(label_text, d)
            except ValidationError as exc:
                raise ValidationError(f"Invalid L3 description at index {idx}: {exc}") from exc

        # Each L3 sentence must differ from L2
        l2_stripped = l2.strip()
        all_identical = all(
            isinstance(s, str) and s.strip() == l2_stripped for s in l3_list
        )
        if all_identical:
            raise ValidationError("All L3 descriptions are identical to L2")

        # L3 sentences must be distinct from each other
        stripped = [s.strip() for s in l3_list if isinstance(s, str)]
        if len(set(stripped)) != len(stripped):
            raise ValidationError("L3 descriptions must be distinct from each other")


class MetadataLogger:
    def __init__(self):
        self._records: List[GenerationRecord] = []

    def record(self, rec: GenerationRecord) -> None:
        self._records.append(rec)

    def to_dict(self) -> Dict[str, Any]:
        return {"records": [r.to_dict() for r in self._records]}

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


class TaskAwareLabelGenerator:
    def __init__(
        self,
        template_store: TemplateStore,
        dataset_config: DatasetTaskTypeConfig,
        llm_generator: Any,
        validation_engine: Optional[ValidationEngine] = None,
        metadata_logger: Optional[MetadataLogger] = None,
    ):
        self._templates = template_store
        self._datasets = dataset_config
        self._llm = llm_generator
        self._validator = validation_engine or ValidationEngine()
        self._logger = metadata_logger or MetadataLogger()

    @property
    def metadata_logger(self) -> MetadataLogger:
        return self._logger

    def build_prompts(self, dataset_id: str, label_text: str) -> Tuple[str, str, str]:
        task_type = self._datasets.get_task_type(dataset_id)
        tmpl = self._templates.get(task_type)

        try:
            prompt_l2 = tmpl.l2.format(dataset_name=dataset_id, label_name=label_text, task_type=task_type)
            prompt_l3 = tmpl.l3.format(dataset_name=dataset_id, label_name=label_text, task_type=task_type)
        except Exception as exc:
            raise GenerationError(
                f"Template substitution failed for label={label_text!r} task_type={task_type!r}: {exc}"
            ) from exc

        return task_type, prompt_l2, prompt_l3

    def generate_for_label(self, dataset_id: str, label_text: str, max_retries: int = 3) -> Tuple[str, List[str]]:
        task_type, prompt_l2, prompt_l3 = self.build_prompts(dataset_id=dataset_id, label_text=label_text)

        last_error = None
        for attempt in range(max_retries):
            try:
                l2 = self._llm.generate(dataset_name=dataset_id, label_name=label_text, prompt_template=prompt_l2)
            except Exception as exc:
                raise GenerationError(
                    f"L2 generation failed for label={label_text!r} task_type={task_type!r}: {exc}"
                ) from exc

            try:
                l3_list = self._llm.generate_multi(dataset_name=dataset_id, label_name=label_text, prompt_template=prompt_l3)
            except Exception as exc:
                raise GenerationError(
                    f"L3 generation failed for label={label_text!r} task_type={task_type!r}: {exc}"
                ) from exc

            try:
                self._validator.validate_l3_list(label_text=label_text, l2=l2, l3_list=l3_list)
            except ValidationError as exc:
                last_error = exc
                import logging
                logging.getLogger(__name__).warning(
                    "Validation failed for label=%r (attempt %d/%d): %s",
                    label_text, attempt + 1, max_retries, exc
                )
                continue  # retry

            # Validation passed
            rec = GenerationRecord(
                generation_id=str(uuid.uuid4()),
                generated_at=datetime.now(timezone.utc).isoformat(),
                dataset=dataset_id,
                task_type=task_type,
                label_text=label_text,
                template_l2=prompt_l2,
                template_l3=prompt_l3,
                l2_description=l2,
                l3_description=json.dumps(l3_list, ensure_ascii=False),
            )
            self._logger.record(rec)
            return l2, l3_list

        raise GenerationError(
            f"Validation failed after {max_retries} attempts for label={label_text!r}: {last_error}"
        )

    def generate_batch(self, dataset_id: str, labels: List[str]) -> Tuple[List[Optional[Dict[str, Any]]], List[Optional[str]]]:
        results: List[Optional[Dict[str, Any]]] = []
        errors: List[Optional[str]] = []

        for label_text in labels:
            try:
                l2, l3_list = self.generate_for_label(dataset_id=dataset_id, label_text=label_text)
                results.append({"l2": l2, "l3": l3_list})
                errors.append(None)
            except Exception as exc:
                results.append(None)
                errors.append(str(exc))

        return results, errors
