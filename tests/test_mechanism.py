"""anchor_shares divides by queries x query-layers: an anchor ahead in every layer for every query has share 1."""
import pytest
import torch

from mtkaudit.mechanism import anchor_shares

N_LAYERS = 32


def test_shares_are_fractions_of_query_layers():
    ahead = torch.zeros((3, 1600), dtype=torch.int16)
    ahead[:, 800] = N_LAYERS                     # first malicious anchor: ahead in every layer, every query
    ahead[0, 801] = N_LAYERS // 2                # second: half the layers, one query of three
    share = anchor_shares(ahead, [0, 1, 2], N_LAYERS)
    assert share.shape == (800,)
    assert share[0] == pytest.approx(1.0)
    assert share[1] == pytest.approx(0.5 / 3)
    assert share[2:].max() == 0


def test_shares_use_only_the_given_rows():
    ahead = torch.zeros((4, 1600), dtype=torch.int16)
    ahead[3, 900] = N_LAYERS
    assert anchor_shares(ahead, [0, 1], N_LAYERS)[100] == 0
    assert anchor_shares(ahead, [3], N_LAYERS)[100] == pytest.approx(1.0)


def test_a_wrong_denominator_is_caught():
    ahead = torch.full((2, 1600), N_LAYERS, dtype=torch.int16)
    with pytest.raises(AssertionError):
        anchor_shares(ahead, [0, 1], N_LAYERS // 2)
