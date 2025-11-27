#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Iterable, List, Tuple


TS_RE = re.compile(r"\d\d:\d\d:\d\d,\d{3}\s+-->\s+\d\d:\d\d:\d\d,\d{3}")
IDX_RE = re.compile(r"^\s*\d+\s*$")
HTML_TAG_RE = re.compile(r"<[^>]+>")
ASS_TAG_RE = re.compile(r"\{[^}]*\}")
BRACKETED_RE = re.compile(r"[\(\[（【<＜][^\)\]）】>＞]{0,80}[\)\]）】>＞]")
MULTI_WS_RE = re.compile(r"\s+")


def has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def has_latin_alpha(s: str) -> bool:
    return any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in s)


def normalize_text(line: str) -> str:
    # strip HTML/ASS tags, bracketed descriptions, repeated punctuation
    s = line.strip()
    s = HTML_TAG_RE.sub(" ", s)
    s = ASS_TAG_RE.sub(" ", s)
    s = BRACKETED_RE.sub(" ", s)
    s = s.replace("\u200b", " ")  # zero-width space
    s = s.replace("\ufeff", " ")  # BOM if present
    s = s.replace("字幕", " ")  # common noise in samples
    # unify hyphens/quotes
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    s = MULTI_WS_RE.sub(" ", s)
    return s.strip().strip("-~#*")


def parse_srt_blocks(text: str) -> List[List[str]]:
    blocks: List[List[str]] = []
    cur: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def parse_ass_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines():
        if raw.startswith("Dialogue:"):
            # ASS: Dialogue: Marked,Start,End,Style,Actor,MarginL,MarginR,MarginV,Effect,Text
            parts = raw.split("Dialogu" + "e:", 1)[-1].split(",", 9)
            if len(parts) == 10:
                text_field = parts[-1]
            else:
                # fallback: after the 9th comma is text
                try:
                    text_field = raw.split(",", 9)[-1]
                except Exception:
                    text_field = raw
            lines.append(text_field)
    return lines


def collect_pairs_from_block(lines: List[str]) -> List[Tuple[str, str]]:
    # remove SRT index/timestamps then normalize
    content = [ln for ln in lines if not TS_RE.search(ln) and not IDX_RE.match(ln)]
    content = [normalize_text(ln) for ln in content if normalize_text(ln)]
    if not content:
        return []
    zh_lines = [ln for ln in content if has_cjk(ln)]
    en_lines = [ln for ln in content if has_latin_alpha(ln) and not has_cjk(ln)]
    if zh_lines and en_lines:
        zh = " ".join(zh_lines)
        en = " ".join(en_lines)
        return [(en, zh)]
    return []


def collect_pairs_from_file(path: str) -> List[Tuple[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return []
    pairs: List[Tuple[str, str]] = []
    if path.lower().endswith(".srt"):
        for block in parse_srt_blocks(text):
            pairs.extend(collect_pairs_from_block(block))
    else:
        # ASS: treat each dialogue line as a block
        for dlg in parse_ass_lines(text):
            pairs.extend(collect_pairs_from_block([dlg]))
    return pairs


def good_pair(en: str, zh: str) -> bool:
    if len(en) < 3 or len(zh) < 2:
        return False
    # token/char length ratio; subtitles往往中短英长，也接受较宽范围
    r = max(0.01, len(zh)) / max(1, len(en))
    if not (0.2 <= r <= 3.0):
        return False
    # drop lines with mostly punctuation/numbers
    if sum(ch.isalpha() for ch in en) < 2:
        return False
    return True


def to_messages(en: str, zh: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "你是专业字幕译员。仅输出中文译文。"},
            {"role": "user", "content": f"将下列英文翻译为中文：{en}"},
            {"role": "assistant", "content": zh},
        ]
    }


def iter_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".srt", ".ass")):
                yield os.path.join(dirpath, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="目录：包含 .srt/.ass")
    ap.add_argument("--output-dir", required=True, help="输出目录，将生成 JSONL 与统计")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--dev-ratio", type=float, default=0.05)
    args = ap.parse_args()

    random.seed(args.seed)

    files = list(iter_files(args.input_dir))
    if not files:
        print(f"[ERR] 未找到字幕文件: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output-dir if False else args.output_dir, exist_ok=True)

    # 收集配对
    pair_set = set()
    pairs: List[Tuple[str, str]] = []
    stats = Counter()

    for i, fp in enumerate(files, 1):
        p = collect_pairs_from_file(fp)
        stats["files_scanned"] += 1
        stats["raw_pairs"] += len(p)
        for en, zh in p:
            en_n = normalize_text(en)
            zh_n = normalize_text(zh)
            if not good_pair(en_n, zh_n):
                continue
            key = (en_n, zh_n)
            if key in pair_set:
                continue
            pair_set.add(key)
            pairs.append((en_n, zh_n))
        if i % 500 == 0:
            print(f"[scan] {i}/{len(files)} files, pairs={len(pairs)}")

    if not pairs:
        print("[ERR] 未提取到任何英中配对。", file=sys.stderr)
        sys.exit(2)

    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * args.train_ratio)
    n_dev = int(n * args.dev_ratio)
    n_test = n - n_train - n_dev

    splits = {
        "train": pairs[:n_train],
        "dev": pairs[n_train:n_train + n_dev],
        "test": pairs[n_train + n_dev:],
    }

    for split, data in splits.items():
        outp = os.path.join(args.output_dir, f"{split}.jsonl")
        with open(outp, "w", encoding="utf-8") as f:
            for en, zh in data:
                f.write(json.dumps(to_messages(en, zh), ensure_ascii=False) + "\n")

    # 统计信息
    report = {
        "files_scanned": stats["files_scanned"],
        "raw_pairs": stats["raw_pairs"],
        "dedup_pairs": len(pairs),
        "train_dev_test": {k: len(v) for k, v in splits.items()},
        "examples": [{"en": pairs[i][0], "zh": pairs[i][1]} for i in range(min(5, len(pairs)))]
    }
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


