import os
import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ==========================================
# 1. Configuration
# ==========================================
# Mistral Small 3 is 24B parameters. Unsloth has pre-quantized 4-bit versions.
MODEL_NAME = "unsloth/mistral-small-24b-instruct-2501-bnb-4bit" 
DATASET_PATH = "raw_training_data.json"
OUTPUT_DIR = "mistral-small-24b-ifc-pma"

max_seq_length = 2048 # Adjust if you need longer context
dtype = None          # Auto-detection
load_in_4bit = True   # REQUIRED for a 24B model to fit on a single GPU (24GB VRAM)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires a CUDA GPU. Run it on a GPU server with enough VRAM for a 24B model.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # ==========================================
    # 2. Load Model & Tokenizer
    # ==========================================
    print(f"Loading {MODEL_NAME} in 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Apply LoRA adapters (we only train ~1-2% of the parameters to save memory)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # Set the chat template for Mistral
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
    )

    # ==========================================
    # 3. Prepare Dataset
    # ==========================================
    def formatting_prompts_func(examples):
        prompts = examples["user_prompt"]
        outputs = examples["expected_code"]
        texts = []

        for prompt, output in zip(prompts, outputs):
            # Format into the standard ChatML / Mistral chat structure
            messages = [
                {"role": "system", "content": "You are an expert database engineer for an IFC Building Information Model. You write perfect PostgreSQL queries."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output}
            ]
            # Apply the tokenizer's chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        return {"text": texts}

    print(f"Loading dataset from {DATASET_PATH}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("raw_training_data.json must contain a non-empty JSON array of training records.")

    required_keys = {"user_prompt", "expected_code"}
    missing_keys = required_keys.difference(raw_data[0].keys())
    if missing_keys:
        raise ValueError(f"Training records are missing required keys: {sorted(missing_keys)}")

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(raw_data)

    # Apply formatting
    dataset = hf_dataset.map(formatting_prompts_func, batched=True)

    # ==========================================
    # 4. Train the Model
    # ==========================================
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, # Increase this to 3 or 4 for better results
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
        ),
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print(trainer_stats)

    # ==========================================
    # 5. Save the Model
    # ==========================================
    print(f"Saving LoRA adapters to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR) # Local saving
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Optional: Export to GGUF (for use with Ollama / LM Studio locally)
    # Uncomment the following block if you want to export to GGUF automatically.
    '''
    print("Exporting to GGUF format...")
    model.save_pretrained_gguf("model_q4_k_m", tokenizer, quantization_method = "q4_k_m")
    print("GGUF export complete. You can now load this into Ollama!")
    '''


if __name__ == "__main__":
    main()
