"""Fix failed label descriptions by regenerating them individually."""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from scripts.generate_label_descriptions import DescriptionGenerator
from src.label_descriptions.task_aware_label_generator import (
    DatasetTaskTypeConfig, TaskAwareLabelGenerator, TemplateStore, ValidationEngine
)

model = os.getenv("DESCRIPTION_MODEL", "openai/gpt-4o-mini")
generator = DescriptionGenerator(model)
template_store = TemplateStore.load("src/label_descriptions/prompt_templates.yaml")
dataset_cfg = DatasetTaskTypeConfig.load("src/label_descriptions/dataset_task_types.yaml")
validator = ValidationEngine(min_words=9, max_words=35)
task_aware = TaskAwareLabelGenerator(
    template_store=template_store,
    dataset_config=dataset_cfg,
    llm_generator=generator,
    validation_engine=validator,
)

# Fix yahoo label 1
yahoo_path = Path("src/label_descriptions/generated_descriptions_yahoo.json")
yahoo = json.loads(yahoo_path.read_text(encoding="utf-8"))
l2, l3 = task_aware.generate_for_label("yahoo_answers_topics", "science and mathematics")
yahoo["yahoo_answers_topics"]["1"] = {"l2": l2, "l3": l3}
yahoo_path.write_text(json.dumps(yahoo, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"yahoo label 1: {l2}")

# Fix 20ng labels 3, 12, 18
ng_path = Path("src/label_descriptions/generated_descriptions_20ng.json")
ng = json.loads(ng_path.read_text(encoding="utf-8"))

for label_id, label_name in [("3", "comp.sys.ibm.pc.hardware"), ("12", "sci.electronics"), ("18", "talk.politics.misc")]:
    l2, l3 = task_aware.generate_for_label("SetFit/20_newsgroups", label_name)
    ng["SetFit/20_newsgroups"][label_id] = {"l2": l2, "l3": l3}
    print(f"20ng label {label_id}: {l2}")

ng_path.write_text(json.dumps(ng, indent=2, ensure_ascii=False), encoding="utf-8")
print("Done.")
