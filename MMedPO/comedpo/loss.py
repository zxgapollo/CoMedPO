"""The unweighted CoMedPO objective in main-paper Equations (11)-(14)."""

import math
import torch
import torch.nn.functional as F

# Each column is a sequence log likelihood, summed over answer tokens only.
BRANCHES = (
    "original_chosen", "background_rejected",
    "factual_chosen", "counterfactual_chosen",
    "factual_rejected", "counterfactual_rejected",
)


def comedpo_loss(policy_logps, reference_logps, beta=0.1, causal_weight=1.0):
    """Return mean total/base/causal losses and per-example preference margins.

    Inputs have shape [batch, 6] in BRANCHES order. beta is eta^{-1} in
    the paper, not the optimizer learning rate. References are always detached.
    Both responses receive a counterfactual correction; no clinical weights
    or length normalization are used.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if not math.isfinite(causal_weight) or causal_weight < 0:
        raise ValueError("causal_weight must be finite and nonnegative")
    if policy_logps.ndim != 2 or policy_logps.shape[1] != len(BRANCHES):
        raise ValueError("Expected policy_logps with shape [batch, 6]")
    if reference_logps.shape != policy_logps.shape or policy_logps.shape[0] == 0:
        raise ValueError("Policy and reference must have matching nonempty shapes")
    ratios = policy_logps.float() - reference_logps.detach().float()
    if not torch.isfinite(ratios).all():
        raise ValueError("Non-finite policy/reference log likelihoods")
    base_margin = ratios[:, 0] - ratios[:, 1]
    causal_margin = (ratios[:, 2] - ratios[:, 3]) - (ratios[:, 4] - ratios[:, 5])
    base = -F.logsigmoid(beta * base_margin).mean()
    causal = -F.logsigmoid(beta * causal_margin).mean()
    return {
        "loss": base + causal_weight * causal,
        "dpo_loss": base,
        "causal_loss": causal,
        "base_margin": base_margin,
        "causal_margin": causal_margin,
    }


def sequence_logps(logits, labels):
    """Sum next-token log probabilities, ignoring prompt/image/padding labels."""
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logits and multimodal-expanded labels must align")
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    if not mask.any(dim=1).all():
        raise ValueError("Every sequence must contain at least one answer token")
    # Cross entropy avoids materializing a second full log-softmax tensor.
    token_nll = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), targets,
        ignore_index=-100, reduction="none",
    )
    return -(token_nll * mask).sum(dim=1)
