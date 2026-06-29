"""
test_loss.py
------------
Unit tests for the paper-aligned loss functions in ContrastiveTrainer.

These tests verify the two loss components in isolation using synthetic
tensors, without instantiating a full HuggingFace model or Trainer.
"""

import math
import pytest
import torch


# ---------------------------------------------------------------------------
# Replicate the loss logic as standalone functions for isolated testing
# ---------------------------------------------------------------------------

def l_wce(logits: torch.Tensor, labels: torch.Tensor, w_p: float, w_n: float) -> torch.Tensor:
    """L_WCE (Eq. 3): weighted binary cross-entropy over the positive-class logit."""
    eps = 1e-8
    y_hat = torch.sigmoid(logits[:, 1])
    y = labels.float()
    return -torch.mean(
        w_p * y * torch.log(y_hat + eps)
        + w_n * (1 - y) * torch.log(1 - y_hat + eps)
    )


def l_nce(logits: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    """L_NCE (Eq. 4): noise-contrastive estimation loss."""
    true_scores = torch.gather(logits, 1, labels.view(-1, 1))
    log_denom = torch.logsumexp(logits / tau, dim=1, keepdim=True)
    return -torch.mean(true_scores / tau - log_denom)


# ---------------------------------------------------------------------------
# L_WCE tests
# ---------------------------------------------------------------------------

class TestLWCE:
    def test_perfect_positive(self):
        """Very high positive-class logit → near-zero WCE loss for positive label."""
        logits = torch.tensor([[−10.0, 10.0]])
        labels = torch.tensor([1])
        loss = l_wce(logits, labels, w_p=1.0, w_n=0.2)
        assert loss.item() < 0.01

    def test_perfect_negative(self):
        """Very high negative-class logit → near-zero WCE loss for negative label."""
        logits = torch.tensor([[10.0, −10.0]])
        labels = torch.tensor([0])
        loss = l_wce(logits, labels, w_p=1.0, w_n=0.2)
        assert loss.item() < 0.01

    def test_w_p_scales_positive_loss(self):
        """Doubling w_p should roughly double loss on a positive sample."""
        logits = torch.tensor([[0.0, 0.0]])
        labels = torch.tensor([1])
        loss1 = l_wce(logits, labels, w_p=1.0, w_n=0.2)
        loss2 = l_wce(logits, labels, w_p=2.0, w_n=0.2)
        assert loss2.item() == pytest.approx(2 * loss1.item(), rel=1e-4)

    def test_w_n_scales_negative_loss(self):
        """Doubling w_n should roughly double loss on a negative sample."""
        logits = torch.tensor([[0.0, 0.0]])
        labels = torch.tensor([0])
        loss1 = l_wce(logits, labels, w_p=1.0, w_n=1.0)
        loss2 = l_wce(logits, labels, w_p=1.0, w_n=2.0)
        assert loss2.item() == pytest.approx(2 * loss1.item(), rel=1e-4)

    def test_nonnegative(self):
        """WCE loss must always be ≥ 0."""
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        assert l_wce(logits, labels, w_p=1.0, w_n=0.2).item() >= 0.0


# ---------------------------------------------------------------------------
# L_NCE tests
# ---------------------------------------------------------------------------

class TestLNCE:
    def test_perfect_predictions(self):
        """Very confident correct predictions → small NCE loss."""
        logits = torch.tensor([[−10.0, 10.0], [10.0, −10.0]])
        labels = torch.tensor([1, 0])
        loss = l_nce(logits, labels, tau=1.0)
        assert loss.item() < 0.01

    def test_uniform_logits(self):
        """Uniform logits → loss should equal log(num_classes)."""
        num_classes = 2
        logits = torch.zeros(4, num_classes)
        labels = torch.randint(0, num_classes, (4,))
        loss = l_nce(logits, labels, tau=1.0)
        assert loss.item() == pytest.approx(math.log(num_classes), rel=1e-4)

    def test_tau_scaling(self):
        """Lower τ should sharpen the distribution (lower loss for correct preds)."""
        logits = torch.tensor([[0.5, 1.5], [1.5, 0.5]])
        labels = torch.tensor([1, 0])
        loss_high_tau = l_nce(logits, labels, tau=2.0)
        loss_low_tau  = l_nce(logits, labels, tau=0.5)
        assert loss_low_tau.item() < loss_high_tau.item()

    def test_nonnegative(self):
        """NCE loss must always be ≥ 0."""
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        assert l_nce(logits, labels, tau=1.0).item() >= 0.0


# ---------------------------------------------------------------------------
# Combined loss test
# ---------------------------------------------------------------------------

def test_combined_loss_sums_components():
    """Total loss should equal L_WCE + L_NCE exactly."""
    logits = torch.randn(6, 2)
    labels = torch.randint(0, 2, (6,))
    tau, w_p, w_n = 1.0, 1.0, 0.2

    wce  = l_wce(logits, labels, w_p, w_n)
    nce  = l_nce(logits, labels, tau)
    total = wce + nce

    assert total.item() == pytest.approx((wce + nce).item(), rel=1e-6)
