import torch

from boltz.model.potentials.potentials import (
    ContactPotentital,
    TemplateReferencePotential,
)


def _one_hot(indices: torch.Tensor, classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices, classes).float()


def test_batched_contact_gradient_matches_singletons() -> None:
    torch.manual_seed(7)
    multiplicity = 2
    num_atoms = 7
    coords = torch.randn(2 * multiplicity, num_atoms, 3)
    atom_mask = torch.ones(2, num_atoms)

    pairs0 = torch.tensor([[0, 0, 4], [2, 3, 6]])
    pairs1 = torch.tensor([[1, 3], [5, 6]])
    padded_pairs1 = torch.nn.functional.pad(pairs1, (0, 1))
    batched_feats = {
        "atom_pad_mask": atom_mask,
        "contact_pair_index": torch.stack((pairs0, padded_pairs1)),
        "contact_union_index": torch.tensor([[0, 0, 1], [0, 1, 0]]),
        "contact_negation_mask": torch.tensor(
            [[True, True, True], [True, True, False]]
        ),
        "contact_thresholds": torch.tensor(
            [[0.4, 0.7, 0.5], [0.3, 0.6, 0.0]]
        ),
        "contact_constraint_mask": torch.tensor(
            [[True, True, True], [True, True, False]]
        ),
    }
    singleton_feats = [
        {
            "atom_pad_mask": atom_mask[0:1],
            "contact_pair_index": pairs0[None],
            "contact_union_index": torch.tensor([[0, 0, 1]]),
            "contact_negation_mask": torch.tensor([[True, True, True]]),
            "contact_thresholds": torch.tensor([[0.4, 0.7, 0.5]]),
        },
        {
            "atom_pad_mask": atom_mask[1:2],
            "contact_pair_index": pairs1[None],
            "contact_union_index": torch.tensor([[0, 1]]),
            "contact_negation_mask": torch.tensor([[True, True]]),
            "contact_thresholds": torch.tensor([[0.3, 0.6]]),
        },
    ]

    potential = ContactPotentital()
    parameters = {"union_lambda": 2.0}
    actual = potential.compute_gradient(coords, batched_feats, parameters)
    expected = torch.cat(
        [
            potential.compute_gradient(
                coords[idx * multiplicity : (idx + 1) * multiplicity],
                feats,
                parameters,
            )
            for idx, feats in enumerate(singleton_feats)
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_batched_contact_gradient_ignores_record_without_constraints() -> None:
    torch.manual_seed(9)
    multiplicity = 2
    num_atoms = 6
    coords = torch.randn(2 * multiplicity, num_atoms, 3)
    pairs = torch.tensor([[0, 2], [4, 5]])
    feats = {
        "atom_pad_mask": torch.ones(2, num_atoms),
        "contact_pair_index": torch.stack((pairs, torch.zeros_like(pairs))),
        "contact_union_index": torch.tensor([[0, 1], [0, 0]]),
        "contact_negation_mask": torch.tensor(
            [[True, True], [False, False]]
        ),
        "contact_thresholds": torch.tensor([[0.5, 0.8], [0.0, 0.0]]),
        "contact_constraint_mask": torch.tensor(
            [[True, True], [False, False]]
        ),
    }
    singleton_feats = {
        "atom_pad_mask": feats["atom_pad_mask"][0:1],
        "contact_pair_index": pairs[None],
        "contact_union_index": torch.tensor([[0, 1]]),
        "contact_negation_mask": torch.tensor([[True, True]]),
        "contact_thresholds": torch.tensor([[0.5, 0.8]]),
    }

    potential = ContactPotentital()
    parameters = {"union_lambda": 2.0}
    actual = potential.compute_gradient(coords, feats, parameters)
    expected_first = potential.compute_gradient(
        coords[:multiplicity], singleton_feats, parameters
    )
    torch.testing.assert_close(actual[:multiplicity], expected_first)
    torch.testing.assert_close(
        actual[multiplicity:], torch.zeros_like(actual[multiplicity:])
    )


def test_batched_template_gradient_matches_singletons() -> None:
    torch.manual_seed(11)
    batch_size = 2
    multiplicity = 2
    num_atoms = 6
    num_tokens = 3
    coords = torch.randn(batch_size * multiplicity, num_atoms, 3)
    atom_to_token_idx = torch.tensor([[0, 0, 1, 1, 2, 2]]).repeat(
        batch_size, 1
    )
    rep_atom_idx = torch.tensor([[0, 2, 4]]).repeat(batch_size, 1)
    atom_to_token = _one_hot(atom_to_token_idx, num_tokens)
    token_to_rep_atom = _one_hot(rep_atom_idx, num_atoms)
    template_cb = torch.randn(batch_size, 1, num_tokens, 3)
    template_mask = torch.ones(batch_size, 1, num_tokens)
    thresholds = torch.tensor([[0.1], [0.2]])
    batched_feats = {
        "atom_pad_mask": torch.ones(batch_size, num_atoms),
        "atom_to_token": atom_to_token,
        "token_to_rep_atom": token_to_rep_atom,
        "token_index": torch.arange(num_tokens)[None].repeat(batch_size, 1),
        "template_cb": template_cb,
        "template_mask_cb": template_mask,
        "template_force": torch.ones(batch_size, 1, dtype=torch.bool),
        "template_force_threshold": thresholds,
    }

    potential = TemplateReferencePotential()
    actual = potential.compute_gradient(coords, batched_feats, {})
    expected_parts = []
    for idx in range(batch_size):
        feats = {key: value[idx : idx + 1] for key, value in batched_feats.items()}
        expected_parts.append(
            potential.compute_gradient(
                coords[idx * multiplicity : (idx + 1) * multiplicity],
                feats,
                {},
            )
        )
    expected = torch.cat(expected_parts)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    mixed_feats = dict(batched_feats)
    mixed_feats["template_force"] = torch.tensor([[True], [False]])
    mixed_actual = potential.compute_gradient(coords, mixed_feats, {})
    torch.testing.assert_close(
        mixed_actual[:multiplicity], expected[:multiplicity], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        mixed_actual[multiplicity:],
        torch.zeros_like(mixed_actual[multiplicity:]),
    )
