import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import json

dataset = load_dataset("Anthropic/hh-rlhf", split="train")
print(f"Dataset size: {len(dataset)}")
print("Dataset features:", dataset.features.keys())
# Load a small subset for local testing
small_dataset = dataset.select(range(1000))

# Load SmolLM3-3B-Instruct model
model_name = "HuggingFaceTB/SmolLM3-3B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Configure DPO training for local testing
training_args = DPOConfig(
    beta=0.1,                           # Preference optimization strength
    learning_rate=5e-7,                 # Lower than SFT
    per_device_train_batch_size=1,      # Small batch for local testing
    gradient_accumulation_steps=4,      # Effective batch size = 4
    max_steps=1000,                       # Very short for testing
    logging_steps=10,
    output_dir="./local_dpo_test",
    report_to="trackio",
)

# Create trainer (but don't train yet - save resources for HF Jobs)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=small_dataset,
    processing_class=tokenizer,
)

print("Local DPO trainer configured successfully!")

trainer.train()