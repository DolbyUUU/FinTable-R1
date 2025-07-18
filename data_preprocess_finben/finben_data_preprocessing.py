#!/usr/bin/env python3
"""
FinBen format to VERL format data preprocessing.

Reads DATASET_CONFIGS from configs.py, loads each HF dataset,
maps to VERL schema, writes Parquet, and optionally copies to HDFS.
"""

import argparse
import logging
import os
import sys
from typing import Dict

# allow importing configs.py from same directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_DIR)
from configs import DATASET_CONFIGS

from datasets import load_dataset, DatasetDict
from verl.utils.hdfs_io import makedirs, copy

# if True, use 'query' when building user prompt, else use 'text'
USE_QUERY = False

def make_prefix(example: Dict, system_instruction: str):
    """
    Build the system + user prompt for a single example.
    Returns a list of {'role': ..., 'content': ...} dicts.
    """
    # Enumerate choices explicitly
    choices_str = ", ".join(f"({i}) {c}" for i, c in enumerate(example["choices"]))

    if USE_QUERY:
        # Use 'query' first, then show 'text' as data
        user_content = (
            f"{example['query']}\n"
            f"Options: {choices_str}\n\n"
            f"Data:\n{example['text']}"
        )
    else:
        # Use only the 'text' field
        user_content = (
            f"{example['text']}\n\n"
            f"Options: {choices_str}"
        )

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user",   "content": user_content},
    ]


def make_map_fn(split_name: str, cfg: Dict[str, str]):
    """
    Returns a function for Dataset.map to convert examples to the VERL schema.
    Uses cfg["ability"] if present, else defaults to 'credit_scoring'.
    """
    def map_fn(example, idx):
        label_idx = example["gold"]
        return {
            "data_source": cfg["data_source"],
            "ability": cfg.get("ability", "credit_scoring"),
            "prompt": make_prefix(example, cfg["system_instruction"]),
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "label_idx": label_idx,
                    "label_str": example["choices"][label_idx],
                }
            },
            "extra_info": {
                "split": split_name,
                "index": idx,
                "id": example["id"],
            }
        }
    return map_fn


def main():
    global USE_QUERY

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Generalized FinBen data preprocessing for VERL."
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="*",
        choices=list(DATASET_CONFIGS.keys()),
        default=None,
        help="Which datasets to process. If omitted, all supported."
    )
    parser.add_argument(
        "--local_dir", type=str, required=True,
        help="Root directory to write subdirectories for each dataset."
    )
    parser.add_argument(
        "--hdfs_dir", type=str, default=None,
        help="Optional HDFS directory to copy all processed data."
    )
    parser.add_argument(
        "--num_proc", type=int, default=4,
        help="Number of parallel processes for dataset.map()."
    )
    parser.add_argument(
        "--use_query", action="store_true",
        help="If set, use the 'query' column (plus Data:text) for user_content; "
             "otherwise use the 'text' column only."
    )

    args = parser.parse_args()

    # Set the global flag based on command-line argument
    USE_QUERY = args.use_query
    logging.info(f"use_query = {USE_QUERY}")

    selected = args.datasets or list(DATASET_CONFIGS.keys())
    logging.info(f"Datasets to process: {selected}")

    root_local = os.path.expanduser(args.local_dir)
    os.makedirs(root_local, exist_ok=True)

    for key in selected:
        cfg = DATASET_CONFIGS[key]
        logging.info(f"Loading {cfg['hf_id']} ...")
        try:
            ds: DatasetDict = load_dataset(cfg["hf_id"])
        except Exception as e:
            logging.error(f"Failed to load {cfg['hf_id']}: {e}")
            continue

        out_dir = os.path.join(root_local, key)
        os.makedirs(out_dir, exist_ok=True)

        # Alias 'valid' or 'val' splits to 'validation' if needed
        for alias in ("valid", "val"):
            if alias in ds and "validation" not in ds:
                ds["validation"] = ds.pop(alias)

        for split in ["train", "validation", "test"]:
            if split not in ds:
                logging.warning(f"  • no split '{split}' in {cfg['hf_id']}, skipping.")
                continue

            part = ds[split]
            n = len(part)
            logging.info(f"  • processing {split} ({n} examples)...")

            mapped = part.map(
                make_map_fn(split, cfg),
                with_indices=True,
                num_proc=args.num_proc,
                remove_columns=part.column_names,
                desc=f"map {key}/{split}"
            )

            out_path = os.path.join(out_dir, f"{split}.parquet")
            mapped.to_parquet(out_path)
            logging.info(f"    → wrote {out_path}")

    if args.hdfs_dir:
        try:
            logging.info(f"Copying {root_local} to HDFS at {args.hdfs_dir} ...")
            makedirs(args.hdfs_dir)
            copy(src=root_local, dst=args.hdfs_dir)
            logging.info("HDFS copy succeeded.")
        except Exception as e:
            logging.error(f"HDFS copy failed: {e}")


if __name__ == "__main__":
    main()