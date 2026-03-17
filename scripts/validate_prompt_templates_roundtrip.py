"""Validate prompt template configuration round-trip integrity.

Usage:
    python -m scripts.validate_prompt_templates_roundtrip
"""

from pathlib import Path

from src.label_descriptions.task_aware_label_generator import TemplateStore


def main() -> None:
    template_path = Path("src/label_descriptions/prompt_templates.yaml")
    store1 = TemplateStore.load(template_path)

    dumped = store1.dumps_yaml()
    tmp_path = Path("src/label_descriptions/_prompt_templates_roundtrip_tmp.yaml")
    tmp_path.write_text(dumped, encoding="utf-8")

    store2 = TemplateStore.load(tmp_path)

    if store1.serialize() != store2.serialize():
        raise SystemExit("Round-trip validation failed: serialized template stores are not equivalent")

    tmp_path.unlink(missing_ok=True)
    print("OK: TemplateStore load -> dump -> load round-trip is equivalent")


if __name__ == "__main__":
    main()
