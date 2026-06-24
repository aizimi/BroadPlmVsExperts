"""data_loaders package

Dataset loader entry points.

Each loader returns a HuggingFace DatasetDict that is:
- cleaned
- mapped to 3 sentiment classes
- tokenized
- formatted for PyTorch Trainer
"""

from .mac import load_mac_tokenized_dataset
from .astd import load_astd_tokenized_dataset
from .labr import load_labr_tokenized_dataset
from .afrisenti import (
    load_afrisenti_ary_tokenized_dataset,
    load_afrisenti_arq_tokenized_dataset,
)
from .arsas import load_arsas_tokenized_dataset
from .hard import load_hard_tokenized_dataset

# ------------------------------------------------------------------ #
# Canonical dataset registry (single source of truth)
# ------------------------------------------------------------------ #

DATASET_REGISTRY = {
    # tweets
    "maccorpus": {
        "loader": load_mac_tokenized_dataset,
        "display_name": "MACcorpus",
        "domain": "tweets",
        "default_max_length": 128,
        "source_type": "csv",
        "default_csv_path": "data/MACcorpus.csv",
        "text_columns": ["tweets", "tweet", "text"],
    },
    "astd": {
        "loader": load_astd_tokenized_dataset,
        "display_name": "ASTD",
        "domain": "tweets",
        "default_max_length": 128,
        "source_type": "hf",
        "hf_dataset_id": "arbml/ASTD",
        "hf_config": None,
        "text_columns": ["text", "tweet", "tweets", "sentence", "content"],
    },
    "arsas": {
        "loader": load_arsas_tokenized_dataset,
        "display_name": "ArSAS",
        "domain": "tweets",
        "default_max_length": 128,
        "source_type": "hf",
        "hf_dataset_id": "arbml/ArSAS",
        "hf_config": None,
        "text_columns": ["tweet", "Tweet_text", "tweets", "text", "sentence", "content"],
    },
    "afrisenti_ary": {
        "loader": load_afrisenti_ary_tokenized_dataset,
        "display_name": "AfriSenti_ARY",
        "domain": "tweets",
        "default_max_length": 128,
        "source_type": "hf",
        "hf_dataset_id": "masakhane/afrisenti",
        "hf_config": "ary",
        "text_columns": ["tweet", "text", "sentence", "content"],
    },
    "afrisenti_arq": {
        "loader": load_afrisenti_arq_tokenized_dataset,
        "display_name": "AfriSenti_ARQ",
        "domain": "tweets",
        "default_max_length": 128,
        "source_type": "hf",
        "hf_dataset_id": "masakhane/afrisenti",
        "hf_config": "arq",
        "text_columns": ["tweet", "text", "sentence", "content"],
    },
    # long reviews
    "labr": {
        "loader": load_labr_tokenized_dataset,
        "display_name": "LABR",
        "domain": "book reviews",
        "default_max_length": 256,
        "source_type": "hf",
        "hf_dataset_id": "mohamedadaly/labr",
        "hf_config": None,
        "text_columns": ["review", "text", "sentence", "content", "tweet", "tweets"],
    },
    "hard": {
        "loader": load_hard_tokenized_dataset,
        "display_name": "HARD",
        "domain": "hotel reviews",
        "default_max_length": 256,
        "source_type": "hf",
        "hf_dataset_id": "Elnagara/hard",
        "hf_config": None,
        "text_columns": ["review", "text", "content", "comments", "comment", "sentence", "body", "tweet", "tweets"],
    },
}

# Aliases → canonical ids
DATASET_ALIASES = {
    "mac": "maccorpus",
    "maccorpus": "maccorpus",
    "astd": "astd",
    "arsas": "arsas",
    "afrisenti_ary": "afrisenti_ary",
    "ary": "afrisenti_ary",
    "afrisenti_arq": "afrisenti_arq",
    "arq": "afrisenti_arq",
    "labr": "labr",
    "hard": "hard",
}


def normalize_dataset_id(dataset_id: str) -> str:
    key = str(dataset_id).strip().lower()
    if key not in DATASET_ALIASES:
        valid = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset id '{dataset_id}'. Valid ids: {valid}")
    return DATASET_ALIASES[key]

__all__ = [
    # loaders
    "load_mac_tokenized_dataset",
    "load_astd_tokenized_dataset",
    "load_labr_tokenized_dataset",
    "load_afrisenti_ary_tokenized_dataset",
    "load_afrisenti_arq_tokenized_dataset",
    "load_arsas_tokenized_dataset",
    "load_hard_tokenized_dataset",
    # registry
    "DATASET_REGISTRY",
    "DATASET_ALIASES",
    "normalize_dataset_id",
]