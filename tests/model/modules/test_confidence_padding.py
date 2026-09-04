import torch

from boltz.model.modules.confidencev2 import (
    _compute_interface_mask,
    _expand_token_atom_logits,
)


def test_interface_mask_ignores_padded_contacts():
    is_contact = torch.ones(1, 3, 3)
    is_different_chain = torch.zeros(1, 3, 3)
    is_different_chain[:, 0, 2] = 1
    is_different_chain[:, 2, 0] = 1
    is_ligand = torch.zeros(1, 3)
    pad_mask = torch.tensor([[1.0, 1.0, 0.0]])

    result = _compute_interface_mask(
        is_contact,
        is_different_chain,
        is_ligand,
        pad_mask,
    )

    assert torch.equal(result, torch.zeros_like(result))


def test_expand_token_atom_logits_handles_variable_lengths_per_batch():
    logits = torch.tensor(
        [
            [[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]],
            [[[30.0], [31.0], [32.0]], [[40.0], [41.0], [42.0]]],
        ]
    )
    atom_to_token = torch.tensor(
        [
            [[1, 0], [1, 0], [0, 0], [0, 0]],
            [[1, 0], [0, 1], [0, 1], [0, 1]],
        ],
        dtype=torch.float32,
    )
    atom_pad_mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32
    )

    result = _expand_token_atom_logits(logits, atom_to_token, atom_pad_mask)

    expected = torch.tensor(
        [[[10.0], [11.0], [0.0], [0.0]], [[30.0], [40.0], [41.0], [42.0]]]
    )
    assert torch.equal(result, expected)
