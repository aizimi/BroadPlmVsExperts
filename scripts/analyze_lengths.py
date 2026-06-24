from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from data_loaders import DATASET_REGISTRY, normalize_dataset_id
from data_loaders.common import DatasetLengthAnalyzer


DEFAULT_TOKENIZERS = {
    "marbert": "UBC-NLP/MARBERTv2",
    "arabert": "aubmindlab/bert-base-arabertv2",
    "darijabert": "SI2M-Lab/DarijaBERT",
    "egybert": "faisalq/EgyBERT",
    "dziribert": "alger-ia/dziribert",
}
DEFAULT_MAC_CSV = "data/MACcorpus.csv"

DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_OUTDIR = "outputs/length_analysis"
DEFAULT_DATASETS = [
    "maccorpus",
    "astd",
    "arsas",
    "afrisenti_ary",
    "afrisenti_arq",
    "labr",
    "hard",
]


def _coerce_texts(ds: Dataset, candidate_columns: Iterable[str]) -> List[str]:
    for col in candidate_columns:
        if col in ds.column_names:
            return ["" if x is None else str(x) for x in ds[col]]
    raise ValueError(f"None of the candidate text columns {list(candidate_columns)} found in {ds.column_names}")


def resolve_dataset_ids(dataset_ids: Sequence[str]) -> List[str]:
    resolved = [normalize_dataset_id(ds) for ds in dataset_ids]
    # preserve order while removing duplicates
    return list(dict.fromkeys(resolved))


def load_raw_texts(dataset_id: str, mac_csv: str) -> List[str]:
    cfg = DATASET_REGISTRY[dataset_id]
    source_type = cfg["source_type"]

    if source_type == "csv":
        csv_path = mac_csv or cfg.get("default_csv_path")
        ds = load_dataset("csv", data_files=csv_path)["train"]
    elif source_type == "hf":
        hf_dataset_id = cfg["hf_dataset_id"]
        hf_config = cfg.get("hf_config")
        ds = load_dataset(hf_dataset_id, hf_config)["train"] if hf_config else load_dataset(hf_dataset_id)["train"]
    else:
        raise ValueError(f"Unsupported source_type '{source_type}' for dataset '{dataset_id}'")

    return _coerce_texts(ds, cfg["text_columns"])


def analyze_all(
    tokenizers: dict[str, str],
    mac_csv: str,
    sample_size: int,
    dataset_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, DatasetLengthAnalyzer]]:
    analyzers: dict[str, DatasetLengthAnalyzer] = {}
    all_rows: list[dict] = []

    resolved_ids = resolve_dataset_ids(dataset_ids)
    cached_texts: dict[str, tuple[dict, list[str]]] = {}

    for dataset_id in resolved_ids:
        cfg = DATASET_REGISTRY[dataset_id]
        texts = load_raw_texts(dataset_id, mac_csv)
        cached_texts[dataset_id] = (cfg, texts)

    for tokenizer_alias, tokenizer_name in tokenizers.items():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        analyzer = DatasetLengthAnalyzer(tokenizer=tokenizer, sample_size=sample_size)
        analyzers[tokenizer_alias] = analyzer

        for dataset_id in resolved_ids:
            cfg, texts = cached_texts[dataset_id]
            row = analyzer.add_dataset(
                dataset_name=cfg["display_name"],
                texts=texts,
                domain=cfg["domain"],
                chosen_max_length=cfg["default_max_length"],
            )
            row["tokenizer_alias"] = tokenizer_alias
            row["tokenizer_name"] = tokenizer_name
            all_rows.append(dict(row))

    detailed_df = pd.DataFrame(all_rows)

    summary_rows: list[dict] = []
    for dataset_id in resolved_ids:
        cfg = DATASET_REGISTRY[dataset_id]
        dataset_name = cfg["display_name"]
        subset = detailed_df[detailed_df["dataset"] == dataset_name].copy()
        if subset.empty:
            continue

        summary_rows.append(
            {
                "dataset": dataset_name,
                "domain": cfg["domain"],
                "n_tokenizers": int(subset["tokenizer_alias"].nunique()),
                "sample_size": int(subset["sample_size"].iloc[0]),
                "median_tokens_min": round(float(subset["median_tokens"].min()), 1),
                "median_tokens_max": round(float(subset["median_tokens"].max()), 1),
                "p95_tokens_min": round(float(subset["p95_tokens"].min()), 1),
                "p95_tokens_max": round(float(subset["p95_tokens"].max()), 1),
                "max_tokens_max": int(subset["max_tokens"].max()),
                "trunc_128_pct_min": round(float(subset["trunc_128_pct"].min()), 2),
                "trunc_128_pct_max": round(float(subset["trunc_128_pct"].max()), 2),
                "trunc_256_pct_min": round(float(subset["trunc_256_pct"].min()), 2),
                "trunc_256_pct_max": round(float(subset["trunc_256_pct"].max()), 2),
                "chosen_max_length": int(cfg["default_max_length"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return detailed_df, summary_df, analyzers


def main() -> None:
    tokenizers = DEFAULT_TOKENIZERS
    mac_csv = str(Path(__file__).parent.parent / DEFAULT_MAC_CSV)
    sample_size = DEFAULT_SAMPLE_SIZE
    dataset_ids = DEFAULT_DATASETS
    outdir = Path(DEFAULT_OUTDIR)

    outdir.mkdir(parents=True, exist_ok=True)

    detailed_df, summary_df, analyzers = analyze_all(
        tokenizers=tokenizers,
        mac_csv=mac_csv,
        sample_size=sample_size,
        dataset_ids=dataset_ids,
    )

    detailed_csv_path = outdir / "token_length_profile_by_tokenizer.csv"
    summary_csv_path = outdir / "token_length_profile_summary.csv"
    summary_md_path = outdir / "token_length_profile_summary.md"
    detailed_json_path = outdir / "token_length_profile_by_tokenizer.json"
    summary_json_path = outdir / "token_length_profile_summary.json"

    detailed_df.to_csv(detailed_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    detailed_json_path.write_text(
        json.dumps(detailed_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_json_path.write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_headers = [
        "Dataset",
        "Domain",
        "#Tok",
        "n",
        "Median min",
        "Median max",
        "P95 min",
        "P95 max",
        "Max",
        "Trunc@128 min%",
        "Trunc@128 max%",
        "Trunc@256 min%",
        "Trunc@256 max%",
        "Chosen max_length",
    ]
    summary_lines = [
        "| " + " | ".join(summary_headers) + " |",
        "| " + " | ".join(["---"] * len(summary_headers)) + " |",
    ]
    for _, row in summary_df.iterrows():
        summary_lines.append(
            "| " + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["domain"]),
                    str(int(row["n_tokenizers"])),
                    str(int(row["sample_size"])),
                    f"{float(row['median_tokens_min']):.1f}",
                    f"{float(row['median_tokens_max']):.1f}",
                    f"{float(row['p95_tokens_min']):.1f}",
                    f"{float(row['p95_tokens_max']):.1f}",
                    str(int(row["max_tokens_max"])),
                    f"{float(row['trunc_128_pct_min']):.2f}",
                    f"{float(row['trunc_128_pct_max']):.2f}",
                    f"{float(row['trunc_256_pct_min']):.2f}",
                    f"{float(row['trunc_256_pct_max']):.2f}",
                    str(int(row["chosen_max_length"])),
                ]
            ) + " |"
        )
    summary_md = "\n".join(summary_lines)
    summary_md_path.write_text(summary_md, encoding="utf-8")

    print("Saved:")
    print(f"  Detailed CSV   : {detailed_csv_path}")
    print(f"  Summary CSV    : {summary_csv_path}")
    print(f"  Summary MD     : {summary_md_path}")
    print(f"  Detailed JSON  : {detailed_json_path}")
    print(f"  Summary JSON   : {summary_json_path}")
    print()
    print(summary_md)


if __name__ == "__main__":
    main()
