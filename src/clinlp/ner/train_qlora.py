"""Workstream B (Week 2): QLoRA fine-tuning script.
Configures and runs SFT (Supervised Fine-Tuning) with PEFT/QLoRA on the synthetic dataset.
Designed to run on GPU/Linux training hosts, falls back gracefully if packages are missing.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

# Optional imports for fine-tuning dependencies
try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer
    HAS_TRAINING_LIBS = True
except ImportError:
    HAS_TRAINING_LIBS = False

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_FILE = REPO_ROOT / "data" / "synthetic" / "train_final.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "synthetic" / "qlora_adapter"


def format_instruction_prompt(sample: dict) -> dict[str, str]:
    """Format a sample into a conversation dict with system/user/assistant roles."""
    system_prompt = (
        "Bạn là một trợ lý AI chuyên gia y tế tiếng Việt. "
        "Nhiệm vụ của bạn là trích xuất toàn bộ thực thể lâm sàng từ văn bản bệnh án "
        "và phân loại chúng thành 5 nhãn (TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC) "
        "cùng với các thuộc tính ngữ cảnh (isNegated, isFamily, isHistorical)."
    )
    
    # Clean entity structure to match training format
    clean_entities = []
    for ent in sample.get("entities", []):
        clean_entities.append({
            "text": ent["text"],
            "type": ent["type"],
            "position": [ent["start"], ent["end"]],
            "assertions": [
                k for k, v in ent.get("assertions", {}).items() if v
            ]
        })
        
    user_message = sample.get("text", "")
    assistant_message = json.dumps({"entities": clean_entities}, ensure_ascii=False)
    
    # We return the chat list format
    return {
        "text": f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_message}<|im_end|>"
    }


def run_qlora_training(
    train_file: str | Path = DEFAULT_TRAIN_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
) -> None:
    """Load base model in 4-bit, configure LoRA, and run supervised fine-tuning."""
    if not HAS_TRAINING_LIBS:
        print("=========================================================================")
        print("WARNING: Training libraries (peft, trl, transformers, datasets, torch) are missing.")
        print("To run QLoRA training, please install requirements using:")
        print("  pip install -e .[llm]")
        print("This script is ready to run on your GPU/Linux training server.")
        print("=========================================================================")
        return

    print(f"Loading dataset from {train_file}...")
    dataset = load_dataset("json", data_files=str(train_file), split="train")
    
    # Format dataset
    dataset = dataset.map(format_instruction_prompt)

    print(f"Loading tokenizer and quantized base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=10,
        num_train_epochs=epochs,
        save_strategy="steps",
        save_steps=100,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting QLoRA training...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {output_dir}...")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Training completed successfully!")


if __name__ == "__main__":
    import json  # Import json inside main for prompt mapping helper
    run_qlora_training()
