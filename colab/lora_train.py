"""QLoRA fine-tune Qwen3-VL-4B-Instruct on 20 KITTI frames for FSD narration.

Mirrors the exact ColabQwenBridge prompt format (single user message: image +
"You are the narration module of an autonomous driving system. Vision facts: ...").
Targets are grounded 2-sentence narrations derived from the real ONNX
detection + lane facts per frame (colab/gen_kit_facts.py).

Usage on the Colab VM:
    python3 lora_train.py train    # QLoRA train -> /content/qwen_lora_adapter
    python3 lora_train.py merge    # fp16 merge  -> /content/qwen_vl_fsd
"""

import gc
import json
import os
import sys
from pathlib import Path

import torch

FRAMES_DIR = Path("/content/frames")
FACTS_JSON = Path("/content/kit_facts.json")
ADAPTER_DIR = "/content/qwen_lora_adapter"
MERGED_DIR = "/content/qwen_vl_fsd"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


# --------------------------------------------------------------------------- data
def _narration_target(facts: dict) -> str:
    """Deterministic 2-sentence target narration grounded in real detections."""
    counts = facts.get("counts", {}) or {}
    lane = int(facts.get("lane_count", 0) or 0)

    if lane:
        s1 = "The road is clear and well-marked, with the drivable area detected."
    else:
        s1 = "The road ahead is visible, though lane markings are not detected."

    nouns = []
    if counts.get("person"):
        nouns.append(f"{counts['person']} pedestrian(s)")
    if counts.get("car"):
        nouns.append(f"{counts['car']} car(s)")
    if counts.get("truck"):
        nouns.append(f"{counts['truck']} truck(s)")
    if counts.get("bicycle"):
        nouns.append(f"{counts['bicycle']} bicycle(s)")
    if counts.get("motorcycle"):
        nouns.append(f"{counts['motorcycle']} motorcycle(s)")
    if counts.get("traffic light"):
        nouns.append("a traffic light")

    if nouns:
        s2 = (f"{', '.join(nouns)} detected nearby. "
              "Maintain safe following distance and be alert for movement into the path.")
    else:
        s2 = "No vehicles or pedestrians detected. Continue at current speed."

    return f"{s1} {s2}"


def _build_examples():
    facts = json.loads(FACTS_JSON.read_text())
    fact_by_frame = {f["frame"]: f for f in facts}
    examples = []
    for fp in sorted(FRAMES_DIR.glob("*.png")):
        fct = fact_by_frame.get(fp.name)
        if fct is None:
            continue
        counts = fct.get("counts", {}) or {}
        parts = [f"{c}: {n}" for c, n in sorted(counts.items(), key=lambda x: -x[1])]
        obj_txt = ", ".join(parts) if parts else "none"
        lane_txt = f"{fct.get('lane_count', 0)} lane(s) fitted"
        user_text = (
            "You are the narration module of an autonomous driving system. "
            f"Vision facts: {obj_txt}. Lane state: {lane_txt}. "
            "Describe the current driving situation in exactly 2 short sentences: "
            "road condition, nearby vehicles/pedestrians, and any hazards. "
            "Answer in plain English, no preamble."
        )
        messages = [
            {"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            {"role": "assistant",
             "content": [{"type": "text", "text": _narration_target(fct)}]},
        ]
        examples.append({"frame": fp.name, "image": str(fp), "messages": messages})
    return examples


# ----------------------------------------------------------------------- collator
def _make_collator(processor):
    def collate(batch):
        msgs_all = [b["messages"] for b in batch]
        msgs_prm = [[m for m in b["messages"] if m["role"] != "assistant"] for b in batch]
        images = [Image.open(b["image"]).convert("RGB") for b in batch]
        texts_all = processor.apply_chat_template(msgs_all, tokenize=False,
                                                  add_generation_prompt=False)
        texts_prm = processor.apply_chat_template(msgs_prm, tokenize=False,
                                                  add_generation_prompt=True)
        out_all = processor(text=texts_all, images=images, return_tensors="pt",
                            padding=True)
        out_prm = processor(text=texts_prm, images=images, return_tensors="pt",
                            padding=True)
        labels = out_all["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        for i in range(len(batch)):
            plen = int((out_prm["attention_mask"][i] != 0).sum())
            labels[i, :plen] = -100
            labels[i][labels[i] == pad_id] = -100
        out_all["labels"] = labels
        return out_all
    return collate


# -------------------------------------------------------------------------- train
def train():
    from torch.utils.data import Dataset
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen3VLForConditionalGeneration,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    examples = _build_examples()
    print(f"[train] {len(examples)} examples")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class _DS(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            return examples[i]

    args = TrainingArguments(
        output_dir="/content/lora_out",
        num_train_epochs=6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_DS(),
        data_collator=_make_collator(processor),
    )
    trainer.train()
    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    print("TRAIN_DONE adapter:", ADAPTER_DIR)


# -------------------------------------------------------------------------- merge
def merge():
    from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration)
    from peft import PeftModel

    print("[merge] loading base fp16...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.merge_and_unload(progressbar=True)
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    AutoProcessor.from_pretrained(ADAPTER_DIR).save_pretrained(MERGED_DIR)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("MERGE_DONE merged:", MERGED_DIR)


if __name__ == "__main__":
    from PIL import Image
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "merge":
        merge()
    else:
        sys.exit(f"unknown command: {cmd}")