import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


@dataclass
class ScriptConfig:
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_split: str = "train"
    dataset_sample_size: Optional[int] = 10_000
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    beta: float = 0.1
    max_prompt_length: int = 512
    max_length: int = 1024
    learning_rate: float = 5e-7
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 1_000
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 500
    save_steps: int = 250
    output_dir: str = "./smollm3-dpo"
    push_to_hub: bool = False
    report_to: str = "none"
    remove_unused_columns: bool = False
    seed: int = 42
    device_map: Optional[str] = "auto"
    load_in_8bit: bool = False
    save_on_finish: bool = False
    cache_dir: str = "./.hf_cache"
    metrics_path: Optional[str] = None


def parse_args() -> ScriptConfig:
    parser = argparse.ArgumentParser(description="SmolLM3 DPO 训练脚本")
    parser.add_argument("--dataset_name", default=ScriptConfig.dataset_name)
    parser.add_argument("--dataset_split", default=ScriptConfig.dataset_split)
    parser.add_argument("--dataset_sample_size", type=int, default=ScriptConfig.dataset_sample_size)
    parser.add_argument("--model_name", default=ScriptConfig.model_name)
    parser.add_argument("--beta", type=float, default=ScriptConfig.beta)
    parser.add_argument("--max_prompt_length", type=int, default=ScriptConfig.max_prompt_length)
    parser.add_argument("--max_length", type=int, default=ScriptConfig.max_length)
    parser.add_argument("--learning_rate", type=float, default=ScriptConfig.learning_rate)
    parser.add_argument("--per_device_train_batch_size", type=int, default=ScriptConfig.per_device_train_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=ScriptConfig.gradient_accumulation_steps)
    parser.add_argument("--max_steps", type=int, default=ScriptConfig.max_steps)
    parser.add_argument("--warmup_steps", type=int, default=ScriptConfig.warmup_steps)
    parser.add_argument("--lr_scheduler_type", default=ScriptConfig.lr_scheduler_type)
    parser.add_argument("--logging_steps", type=int, default=ScriptConfig.logging_steps)
    parser.add_argument("--save_steps", type=int, default=ScriptConfig.save_steps)
    parser.add_argument("--output_dir", default=ScriptConfig.output_dir)
    parser.add_argument("--report_to", default=ScriptConfig.report_to)
    parser.add_argument("--seed", type=int, default=ScriptConfig.seed)
    parser.add_argument("--device_map", default=ScriptConfig.device_map)
    parser.add_argument("--save_on_finish", action="store_true", default=ScriptConfig.save_on_finish)
    parser.add_argument("--no_save_on_finish", action="store_false", dest="save_on_finish")
    parser.add_argument("--push_to_hub", action="store_true", default=ScriptConfig.push_to_hub)
    parser.add_argument("--no_push_to_hub", action="store_false", dest="push_to_hub")
    parser.add_argument("--remove_unused_columns", action="store_true", default=ScriptConfig.remove_unused_columns)
    parser.add_argument("--keep_unused_columns", action="store_false", dest="remove_unused_columns")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=ScriptConfig.gradient_checkpointing)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--bf16", action="store_true", default=ScriptConfig.bf16)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")
    parser.add_argument("--fp16", action="store_true", default=ScriptConfig.fp16)
    parser.add_argument("--no_fp16", action="store_false", dest="fp16")
    parser.add_argument("--load_in_8bit", action="store_true", default=ScriptConfig.load_in_8bit)
    parser.add_argument("--no_load_in_8bit", action="store_false", dest="load_in_8bit")
    parser.add_argument("--cache_dir", default=ScriptConfig.cache_dir)
    parser.add_argument("--metrics_path", default=ScriptConfig.metrics_path)
    args = parser.parse_args()
    return ScriptConfig(**vars(args))


def build_trainer(config: ScriptConfig) -> DPOTrainer:
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        config.dataset_name, split=config.dataset_split, cache_dir=str(cache_dir)
    )
    if config.dataset_sample_size:
        sample_size = min(len(dataset), config.dataset_sample_size)
        dataset = dataset.select(range(sample_size))

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, cache_dir=str(cache_dir)
    )
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {}
    if config.device_map:
        model_kwargs["device_map"] = config.device_map
    if config.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    if config.bf16:
        model_kwargs["dtype"] = torch.bfloat16
    if config.fp16:
        model_kwargs["dtype"] = torch.float16
    model_kwargs.setdefault("low_cpu_mem_usage", True)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, cache_dir=str(cache_dir), **model_kwargs
    )

    dpo_args = DPOConfig(
        beta=config.beta,
        max_prompt_length=config.max_prompt_length,
        max_length=config.max_length,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        push_to_hub=config.push_to_hub,
        report_to=config.report_to,
        remove_unused_columns=config.remove_unused_columns,
        seed=config.seed,
    )

    return DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def run_training(config: ScriptConfig) -> Dict[str, Any]:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer = build_trainer(config)

    start = time.perf_counter()
    train_output = trainer.train()
    wall_time = time.perf_counter() - start

    metrics: Dict[str, Any] = {}
    if train_output is not None and train_output.metrics:
        metrics.update(train_output.metrics)
    metrics["wall_clock_runtime"] = wall_time
    if torch.cuda.is_available():
        metrics["max_memory_bytes"] = torch.cuda.max_memory_allocated()

    if config.save_on_finish:
        trainer.save_model()
        trainer.save_state()

    if config.metrics_path:
        metrics_path = Path(config.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

    return metrics


def main() -> None:
    config = parse_args()
    print("当前训练配置:", asdict(config))
    print("开始 DPO 训练...")
    metrics = run_training(config)
    print("训练完成，关键指标:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
