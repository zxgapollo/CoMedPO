"""CPU regression tests for the paper objective and paired visual conditions."""

import json
import math
import numpy as np
from PIL import Image
import pytest
import torch

from MMedPO.comedpo.loss import comedpo_loss, sequence_logps
from MMedPO.comedpo.data import CoMedPODataset


def test_paper_formula_and_symmetric_gradients():
    policy = torch.tensor([[2., 1., 5., 3., 4., 1.]], requires_grad=True)
    reference = torch.tensor([[1., 1., 2., 1., 1., 0.]], requires_grad=True)
    result = comedpo_loss(policy, reference, beta=0.5, causal_weight=0.75)
    # Base margin = 1, causal margin = (3-2)-(3-1) = -1.
    expected = math.log1p(math.exp(-0.5)) + 0.75 * math.log1p(math.exp(0.5))
    assert result["loss"].item() == pytest.approx(expected)
    result["loss"].backward()
    assert reference.grad is None
    assert torch.equal(torch.sign(policy.grad), torch.tensor([[-1., 1., -1., 1., 1., -1.]]))


def test_identity_reference_and_lambda_zero():
    policy = torch.randn(3, 6, requires_grad=True)
    result = comedpo_loss(policy, policy.detach(), causal_weight=1)
    assert result["loss"].item() == pytest.approx(2 * math.log(2))
    result = comedpo_loss(policy, torch.zeros_like(policy), causal_weight=0)
    assert torch.equal(result["loss"], result["dpo_loss"])
    result["loss"].backward()
    assert torch.count_nonzero(policy.grad[:, 2:]) == 0


def test_causal_cancellation_and_numerical_stability():
    policy = torch.tensor([[1e4, -1e4, 20., 20., 40., 40.]], requires_grad=True)
    result = comedpo_loss(policy, torch.zeros_like(policy))
    assert result["causal_loss"].item() == pytest.approx(math.log(2))
    result["loss"].backward()
    assert torch.isfinite(policy.grad).all()


def test_masking_shift_and_sum():
    logits = torch.zeros(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, -100], [-100, 1, 2, 3, 4]])
    logps = sequence_logps(logits, labels)
    assert torch.allclose(logps, -torch.tensor([2., 4.]) * math.log(7))
    logps.sum().backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    with pytest.raises(ValueError, match="answer token"):
        sequence_logps(logits, torch.full_like(labels, -100))


@pytest.mark.parametrize("kwargs", [{"beta": 0}, {"beta": float("nan")}, {"causal_weight": -1}])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        comedpo_loss(torch.zeros(1, 6), torch.zeros(1, 6), **kwargs)


def make_data(tmp_path):
    Image.new("RGB", (8, 6), (100, 150, 200)).save(tmp_path / "image.png")
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:4, 3:5] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    row = dict(image="image.png", lesion_mask="mask.png", question="Is there a lesion?",
               chosen="Yes", rejected="No", weighted_score=99)
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([row]))
    return path, row


def test_six_branches_share_background_and_answers(tmp_path):
    path, _ = make_data(tmp_path)
    dataset = CoMedPODataset(path)
    rows = dataset[0]
    original, background, factual, counterfactual = [np.asarray(rows[i][2]) for i in range(4)]
    assert (background[2:4, 3:5] == 0).all()
    assert np.array_equal(background[0], original[0])
    assert np.array_equal(factual[:, :8], background)
    assert np.array_equal(counterfactual[:, :8], background)
    assert np.array_equal(factual[:, 8:], original)
    assert not np.array_equal(counterfactual[:, 8:], original)
    assert np.array_equal(counterfactual, np.asarray(dataset[0][3][2]))
    assert rows[2][0] == rows[3][0] == rows[4][0] == rows[5][0]
    assert rows[2][1] == rows[3][1] == "Yes"
    assert rows[4][1] == rows[5][1] == "No"
    assert rows[2][2] is rows[4][2] and rows[3][2] is rows[5][2]


def test_missing_intervention_and_mismatched_prompt_fail(tmp_path):
    path, row = make_data(tmp_path)
    del row["lesion_mask"]
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="background_image or lesion_mask"):
        CoMedPODataset(path)
    row["lesion_mask"] = "mask.png"
    row["conversations"] = [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]
    row["rejected_conversations"] = [{"from": "human", "value": "Other"}, {"from": "gpt", "value": "B"}]
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="same question"):
        CoMedPODataset(path)


def test_legacy_data_requires_intervention_but_ignores_weight(tmp_path):
    path, row = make_data(tmp_path)
    row["conversations"] = [{"from": "human", "value": "<image>\nQ"}, {"from": "gpt", "value": "A"}]
    row["rejected_conversations"] = [{"from": "human", "value": "<image>\nQ"}, {"from": "gpt", "value": "B"}]
    path.write_text(json.dumps([row]))
    data = CoMedPODataset(path)[0]
    assert data[0][:2] == ("Q", "A")
    assert data[1][:2] == ("Q", "B")


def test_precomputed_composites(tmp_path):
    path, row = make_data(tmp_path)
    data = CoMedPODataset(path)[0]
    data[2][2].save(tmp_path / "factual.png")
    data[3][2].save(tmp_path / "counterfactual.png")
    row.update(factual_image="factual.png", counterfactual_image="counterfactual.png")
    path.write_text(json.dumps([row]))
    reloaded = CoMedPODataset(path, noise_std=0.5)[0]
    assert np.array_equal(np.asarray(data[3][2]), np.asarray(reloaded[3][2]))
