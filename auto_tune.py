import json
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from smollm3_dpo import ScriptConfig


REPO_ROOT = Path(__file__).resolve().parent


def build_search_space() -> List[Dict[str, int]]:
    combos = [
        {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 6},
        {"per_device_train_batch_size": 3, "gradient_accumulation_steps": 5},
        {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 4},
        {"per_device_train_batch_size": 5, "gradient_accumulation_steps": 3},
        {"per_device_train_batch_size": 6, "gradient_accumulation_steps": 3},
        {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 2},
    ]
    for combo in combos:
        combo["effective_batch_size"] = (
            combo["per_device_train_batch_size"] * combo["gradient_accumulation_steps"]
        )
    return combos


def prepare_base_config() -> ScriptConfig:
    return ScriptConfig(
        dataset_sample_size=256,
        max_steps=6,
        logging_steps=3,
        save_steps=1_000,
        warmup_steps=0,
        max_length=640,
        push_to_hub=False,
        report_to="none",
        save_on_finish=False,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
    )


def trial_name(per_device_batch: int, grad_accum: int) -> str:
    return f"bs{per_device_batch}_ga{grad_accum}"


def run_trial(
    base_config: ScriptConfig,
    per_device_batch: int,
    grad_accum: int,
) -> Dict:
    output_dir = REPO_ROOT / "sweeps" / trial_name(per_device_batch, grad_accum)
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_config = replace(
        base_config,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        output_dir=str(output_dir),
        metrics_path=str(metrics_path),
    )

    cmd = [
        sys.executable,
        str(REPO_ROOT / "smollm3_dpo.py"),
        "--dataset_name",
        trial_config.dataset_name,
        "--dataset_split",
        trial_config.dataset_split,
        "--dataset_sample_size",
        str(trial_config.dataset_sample_size),
        "--model_name",
        trial_config.model_name,
        "--beta",
        str(trial_config.beta),
        "--max_prompt_length",
        str(trial_config.max_prompt_length),
        "--max_length",
        str(trial_config.max_length),
        "--learning_rate",
        str(trial_config.learning_rate),
        "--per_device_train_batch_size",
        str(trial_config.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(trial_config.gradient_accumulation_steps),
        "--max_steps",
        str(trial_config.max_steps),
        "--warmup_steps",
        str(trial_config.warmup_steps),
        "--lr_scheduler_type",
        trial_config.lr_scheduler_type,
        "--logging_steps",
        str(trial_config.logging_steps),
        "--save_steps",
        str(trial_config.save_steps),
        "--output_dir",
        str(trial_config.output_dir),
        "--report_to",
        trial_config.report_to,
        "--seed",
        str(trial_config.seed),
        "--cache_dir",
        str(trial_config.cache_dir),
        "--metrics_path",
        str(trial_config.metrics_path),
    ]

    if trial_config.push_to_hub:
        cmd.append("--push_to_hub")
    else:
        cmd.append("--no_push_to_hub")
    if trial_config.remove_unused_columns:
        cmd.append("--remove_unused_columns")
    else:
        cmd.append("--keep_unused_columns")
    if trial_config.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    else:
        cmd.append("--no_gradient_checkpointing")
    if trial_config.bf16:
        cmd.append("--bf16")
    else:
        cmd.append("--no_bf16")
    if trial_config.fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--no_fp16")
    if trial_config.load_in_8bit:
        cmd.append("--load_in_8bit")
    else:
        cmd.append("--no_load_in_8bit")
    if trial_config.save_on_finish:
        cmd.append("--save_on_finish")
    else:
        cmd.append("--no_save_on_finish")
    if trial_config.device_map is not None:
        cmd.extend(["--device_map", trial_config.device_map])

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"\n===== 运行试验 {trial_name(per_device_batch, grad_accum)} =====")
    print("命令:", " ".join(cmd))

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    result: Dict = {
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }

    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(completed.stdout)
        if completed.stderr:
            fp.write("\n=== STDERR ===\n")
            fp.write(completed.stderr)

    if completed.returncode == 0 and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fp:
            metrics = json.load(fp)
        result.update(metrics)
        result["status"] = "success"
    else:
        if "CUDA out of memory" in result["stderr"] or "CUDA out of memory" in result["stdout"]:
            result["status"] = "oom"
        else:
            result["status"] = "error"

    return result


def main() -> None:
    base_config = prepare_base_config()
    search_space = build_search_space()

    results: List[Dict] = []
    for combo in search_space:
        try:
            res = run_trial(
                base_config,
                combo["per_device_train_batch_size"],
                combo["gradient_accumulation_steps"],
            )
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            results.append(
                {
                    "per_device_train_batch_size": combo["per_device_train_batch_size"],
                    "gradient_accumulation_steps": combo["gradient_accumulation_steps"],
                    "status": "error",
                    "error": "".join(traceback.format_exception(exc)).strip(),
                }
            )

    print("\n====== 试验结果汇总 ======")
    print(
        "组合\t状态\treturncode\tmax_memory_GB\ttrain_runtime\ttrain_samples_per_second"
    )
    for item in results:
        max_mem = (
            round(item.get("max_memory_bytes", 0) / 1024 / 1024 / 1024, 2)
            if item.get("max_memory_bytes")
            else "-"
        )
        print(
            f"{trial_name(item['per_device_train_batch_size'], item['gradient_accumulation_steps'])}\t"
            f"{item.get('status')}\t"
            f"{item.get('returncode')}\t"
            f"{max_mem}\t"
            f"{item.get('train_runtime', '-')}\t"
            f"{item.get('train_samples_per_second', '-')}"
        )

    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        best = max(
            successful,
            key=lambda r: (r.get("train_samples_per_second", 0.0)),
        )
        print("\n推荐配置:")
        print(
            f"per_device_train_batch_size={best['per_device_train_batch_size']}, "
            f"gradient_accumulation_steps={best['gradient_accumulation_steps']}"
        )
    else:
        print("\n未找到成功的配置，请检查错误日志。")


if __name__ == "__main__":
    main()
