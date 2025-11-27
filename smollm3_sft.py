from re import split
from numpy import dtype
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import trackio as wandb

# Initialize experiment tracking
wandb.init(project="smollm3-sft", name="smollm3-sft-001")

model_name = "HuggingFaceTB/SmolLM3-3B-Base"
instruct_model_name = "HuggingFaceTB/SmolLM3-3B"
# Load SmolLM3 base model
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)

if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load SmolTalk2 dataset
dataset = load_dataset("HuggingFaceTB/smoltalk2", "SFT")
train_dataset =dataset["smoltalk_everyday_convs_reasoning_Qwen3_32B_think"].select(range(1000))

def format_chat_template(example):
    """Format the messages using the chat template"""
    if "messages" in example:
        # SmolTalk2 format
        messages = example["messages"]
    else:
        # Custom format - adapt as needed
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]
    
    # Apply chat template
    text = instruct_tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# Apply formatting
formatted_dataset = train_dataset.map(format_chat_template)
formatted_dataset = formatted_dataset.remove_columns(
    [col for col in formatted_dataset.column_names if col != "text"]
)

# Configure training with Trackio integration
config = SFTConfig(
    output_dir="./smollm3-finetuned",
    dataset_text_field="text",
    max_length=2024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    dataloader_num_workers=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    bf16=True,
    max_steps=1000,
    report_to="trackio",  # Enable Trackio logging
)

# Train!
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=config,
)
trainer.train()
