"""NLI scoring (DeBERTa-v3 MNLI/FEVER/ANLI) for input-text consistency S_x(c) and for
validating constructed equivalence labels. Runs on CPU (tiny model, dynamic shapes)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ("entailment", "neutral", "contradiction")


class NLI:
    def __init__(self, model_id: str, max_length: int = 512, premise_tail_chars: int = 1500):
        self.tok: Any = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.order = [
            next(i for i, lab in id2label.items() if lab.startswith(name[:6])) for name in LABELS
        ]
        self.max_length = max_length
        self.premise_tail_chars = premise_tail_chars

    def _premise(self, text: str) -> str:
        # keep the END of long contexts (closest to the extraction position)
        return text if len(text) <= self.premise_tail_chars else text[-self.premise_tail_chars :]

    @torch.no_grad()
    def probs(self, premises: list[str], hypotheses: list[str], batch_size: int = 16) -> np.ndarray:
        """-> [N,3] probabilities ordered (entailment, neutral, contradiction)."""
        assert len(premises) == len(hypotheses)
        out = np.zeros((len(premises), 3), dtype=np.float32)
        for s in range(0, len(premises), batch_size):
            p = [self._premise(x) for x in premises[s : s + batch_size]]
            h = hypotheses[s : s + batch_size]
            enc = self.tok(
                p,
                h,
                return_tensors="pt",
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
            )
            pr = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.order]
            out[s : s + len(p)] = pr.numpy()
        return out


def support_x(probs: np.ndarray) -> np.ndarray:
    """S_x = P(entail) - P(contradict)."""
    return probs[:, 0] - probs[:, 2]
