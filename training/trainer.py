# training/trainer.py
from __future__ import annotations

import os
import warnings
from typing import Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from training.metrics import build_compute_metrics
from datasets import DatasetDict

def set_global_seed(seed: int) -> None:
    """Best-effort global seeding across python/numpy/torch."""
    # PYTHONHASHSEED must be set in the shell before the interpreter starts.
    # Setting it here has no effect — the interpreter reads it only at startup.
    # Use run_experiment.sh which exports PYTHONHASHSEED before launching Python.

    # Stronger CUDA determinism (may impact performance). Must be set before CUDA kernels run.
    # See PyTorch reproducibility docs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        # Reduce numeric variability (esp. on Ampere+ GPUs)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Enforce deterministic algorithms on CUDA for reproducibility.
        # On Apple MPS this can severely reduce performance, so skip it there.
        try:
            if torch.cuda.is_available():
                torch.use_deterministic_algorithms(True)
        except Exception as e:
            warnings.warn(
                f"torch.use_deterministic_algorithms(True) failed: {e}. "
                "Some CUDA operations may be non-deterministic. "
                "Reproducibility cannot be guaranteed.",
                RuntimeWarning,
                stacklevel=2,
            )
    except Exception as e:
        warnings.warn(
            f"Seed initialisation block failed: {e}. "
            "Reproducibility settings may be incomplete.",
            RuntimeWarning,
            stacklevel=2,
        )


def is_hf_checkpoint_dir(path: str) -> bool:
    if not path or (not os.path.isdir(path)):
        return False
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return False
    weight_files = (
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "flax_model.msgpack",
    )
    return any(os.path.isfile(os.path.join(path, wf)) for wf in weight_files)


def resolve_model_path(base_name: str, ckpt_dir: str) -> str:
    return ckpt_dir if is_hf_checkpoint_dir(ckpt_dir) else base_name

def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers deterministically."""
    import random
    import numpy as np

    # `torch.initial_seed()` is set from the DataLoader generator.
    try:
        import torch
        worker_seed = torch.initial_seed() % 2 ** 32
    except Exception:
        worker_seed = 0

    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DeterministicTrainer(Trainer):
    """Trainer that enforces deterministic DataLoader worker seeding."""

    def __init__(self, *args, seed: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._dl_seed = int(seed)

    def _seeded_generator(self):
        import torch
        g = torch.Generator()
        g.manual_seed(self._dl_seed)
        return g

    def get_train_dataloader(self):
        dl = super().get_train_dataloader()
        dl.worker_init_fn = seed_worker
        try:
            dl.generator = self._seeded_generator()
        except Exception:
            pass
        return dl

    def get_eval_dataloader(self, eval_dataset=None):
        dl = super().get_eval_dataloader(eval_dataset=eval_dataset)
        dl.worker_init_fn = seed_worker
        try:
            dl.generator = self._seeded_generator()
        except Exception:
            pass
        return dl

    def get_test_dataloader(self, test_dataset):
        dl = super().get_test_dataloader(test_dataset)
        dl.worker_init_fn = seed_worker
        try:
            dl.generator = self._seeded_generator()
        except Exception:
            pass
        return dl


class WeightedTrainer(DeterministicTrainer):
    """Trainer with optional class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        if self.class_weights is None:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
        else:
            weight = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def build_trainer(
        ds_tok: DatasetDict,
        base_name: str,
        output_dir: str,
        ckpt_dir: str,
        num_labels: int,
        seed: int,
        tok_name: str,
        class_weights: torch.Tensor | None = None,
) -> Tuple[Trainer, AutoModelForSequenceClassification]:
    set_global_seed(seed)
    set_seed(seed)

    resolved = resolve_model_path(base_name=base_name, ckpt_dir=ckpt_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        resolved,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    compute_metrics = build_compute_metrics()

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        seed=seed,
        data_seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
        seed=seed,
        class_weights=class_weights,
    )

    return trainer, model


def _clear_intermediate_checkpoints(output_dir: str) -> None:
    """Delete any checkpoint-* subdirs left by a prior interrupted run."""
    import shutil
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-") and name[len("checkpoint-"):].isdigit():
            path = os.path.join(output_dir, name)
            shutil.rmtree(path, ignore_errors=True)
            print(f"[INFO] Removed stale intermediate checkpoint: {path}")


def train_or_load_best(trainer: Trainer, ckpt_dir: str) -> AutoModelForSequenceClassification:
    if is_hf_checkpoint_dir(ckpt_dir):
        print("[OK] Checkpoint found. Skipping training and loading fine-tuned weights.")
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        trainer.model = model
        trainer.callback_handler.callbacks.clear()
        return model

    _clear_intermediate_checkpoints(trainer.args.output_dir)
    trainer.train()
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer.save_model(ckpt_dir)
    return trainer.model
