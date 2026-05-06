"""
Build chemistry priors used by both V7 and V8.

These are task-independent priors derived from chemistry / physics common sense,
NOT from any trained model output. Reviewer-defensible.

Output: causal_model/prep_data/chemistry_priors.json
"""
import json
from pathlib import Path

PREP = Path("/home/dell/autodl-tmp/lorafair/fairchem/causal_model/prep_data")
PREP.mkdir(parents=True, exist_ok=True)


PRIORS = {
    "description": (
        "Chemistry priors used as constraints in V7/V8 causal-embedded training. "
        "All priors are model-independent — derived from periodic table / "
        "functional group chemistry / extensive thermodynamics."
    ),
    "version": "1.0",
    "valid_for": ["V7", "V8"],

    # ──────────────────────────────────────────────────────────
    # 1. Cooperative-zero prior (used in L_coop_zero)
    # ──────────────────────────────────────────────────────────
    "cooperative_zero": {
        "rule": "cooperative function ψ(n_h2o, n_co2) must equal 0 when either gas is absent",
        "constraints": [
            {"condition": {"n_h2o": 0}, "expected_value": 0.0,
             "rationale": "no H2O, no water bridge possible"},
            {"condition": {"n_co2": 0}, "expected_value": 0.0,
             "rationale": "no CO2, no co-adsorption synergy"},
        ],
        "loss_form": "MSE(ψ(0, c), 0) + MSE(ψ(h, 0), 0)",
        "weight_lambda": 1.0,
    },

    # ──────────────────────────────────────────────────────────
    # 2. Energy extensivity (used in L_extensive)
    # ──────────────────────────────────────────────────────────
    "energy_extensive": {
        "rule": "predicted energy should scale roughly linearly with N_atoms (extensive quantity)",
        "loss_form": "weak penalty: MSE(E_pred / N_atoms, E_per_atom_mean)",
        "weight_lambda": 0.01,
        "rationale": "binding energy is per molecule but framework size matters for normalization",
    },

    # ──────────────────────────────────────────────────────────
    # 3. Cell volume decorrelation (used in L_decorrel)
    # ──────────────────────────────────────────────────────────
    "cell_volume_decorrel": {
        "rule": "predicted energy should not be directly determined by raw cell volume",
        "loss_form": "MSE(corr(E_pred, cell_volume), 0)",
        "weight_lambda": 0.1,
        "rationale": "different MOFs with similar volume can have very different binding",
    },

    # ──────────────────────────────────────────────────────────
    # 4. Functional group definitions (used in V8 motif encoder)
    # ──────────────────────────────────────────────────────────
    "functional_groups": {
        "carboxylate": {
            "smarts": "[CX3](=O)[O-,OH]",
            "min_atoms": 3,
            "donor_type": "O",
            "expected_binding": "H-bond donor/acceptor",
        },
        "pyridine_n": {
            "smarts": "[n]1ccccc1",
            "min_atoms": 6,
            "donor_type": "N",
            "expected_binding": "Lewis base / metal coordination",
        },
        "imidazolate": {
            "smarts": "[n]1ccnc1",
            "min_atoms": 5,
            "donor_type": "N",
            "expected_binding": "metal coordination",
        },
        "aromatic_C6": {
            "smarts": "[c]1ccccc1",
            "min_atoms": 6,
            "donor_type": "none",
            "expected_binding": "spacer (no direct binding)",
        },
    },

    # ──────────────────────────────────────────────────────────
    # 5. Element categorization (used in atom_type supervision)
    # ──────────────────────────────────────────────────────────
    "element_categories": {
        "transition_metals_first_row": {
            "Z": [22, 23, 24, 25, 26, 27, 28, 29, 30],
            "symbols": ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
            "binding_role": "metal node",
        },
        "transition_metals_second_row": {
            "Z": [40, 41, 42, 44, 45, 46, 47],
            "symbols": ["Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag"],
            "binding_role": "metal node",
        },
        "lanthanides": {
            "Z_range": [57, 71],
            "binding_role": "high-coordination metal node",
        },
        "main_group_metals": {
            "symbols": ["Li", "Na", "K", "Mg", "Ca", "Sr", "Ba", "Al", "Ga"],
            "binding_role": "metal node",
        },
        "ligand_atoms_donor": {
            "symbols": ["O", "N", "S"],
            "binding_role": "coordinator",
        },
        "ligand_atoms_skeleton": {
            "symbols": ["C", "H"],
            "binding_role": "framework skeleton",
        },
    },

    # ──────────────────────────────────────────────────────────
    # 6. Coordination chemistry rules (V8 only; informs site_encoder)
    # ──────────────────────────────────────────────────────────
    "coordination_rules": {
        "open_metal_site_definition": (
            "metal atom with coordination number lower than typical, "
            "providing accessible empty coordination site for guest binding"
        ),
        "typical_cn_by_metal": {
            "Cu": 4, "Zn": 4, "Cd": 6, "Co": 6, "Ni": 6,
            "Mn": 6, "Fe": 6, "Mg": 6, "Al": 6,
            "La": 9, "Ce": 9, "Eu": 9, "Tb": 8,
        },
    },

    # ──────────────────────────────────────────────────────────
    # 7. Gas mode environment definition (used in L_inv / IRM)
    # ──────────────────────────────────────────────────────────
    "environment_definition": {
        "rule": "split samples into 3 environments by gas mode",
        "environments": {
            "pure_CO2": {"condition": {"n_h2o": 0, "n_co2": ">0"}},
            "pure_H2O": {"condition": {"n_co2": 0, "n_h2o": ">0"}},
            "mixed":    {"condition": {"n_h2o": ">0", "n_co2": ">0"}},
        },
        "rationale": "different gas modes activate different binding mechanisms",
    },

    # ──────────────────────────────────────────────────────────
    # 8. Loss weight schedule (recommended starting points)
    # ──────────────────────────────────────────────────────────
    "loss_weights_recommended": {
        "stage_1": {
            "L_energy": 1.0,
            "L_atom_type": 0.1,
            "L_extensive": 0.01,
        },
        "stage_2": {
            "L_energy": 1.0,
            "L_atom_type": 0.05,
            "L_inv": 0.5,
            "L_motif_consistency": 0.1,
            "L_decorrel": 0.05,
        },
        "stage_3_v7": {
            "L_pred": 1.0,
            "L_inv": 0.5,
            "L_S": 1.0,
            "L_coop": 0.5,
            "L_prior": "0.5 → 0.1 (decay over first 5 epochs)",
            "L_reg": 0.01,
        },
        "stage_3_v8": {
            "L_pred": 1.0,
            "L_inv": 0.5,
            "L_coop_zero": 1.0,
            "L_extensive": 0.01,
            "L_decorrel": 0.1,
        },
    },
}


def main():
    out = PREP / "chemistry_priors.json"
    out.write_text(json.dumps(PRIORS, indent=2, ensure_ascii=False))
    print(f"Saved: {out}")
    print(f"Bytes: {out.stat().st_size}")
    print(f"Top-level keys: {list(PRIORS.keys())}")


if __name__ == "__main__":
    main()
