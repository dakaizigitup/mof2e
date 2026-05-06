"""Parallel entry point that loads causal_model BEFORE calling fairchem main.

Use this instead of main.py when running V7/V8 yml configs.

Original main.py is unchanged — running with main.py still works for legacy
yml's like eSCN.yml / eSCN_global.yml / EquiformerV2 yml's.

Usage:
    python fairchem/causal_main.py \
        --config-yml=fairchem/causal_model/config_causal/v7/eSCN_v7_s1.yml \
        --mode=train
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # Must import irm_trainer first to keep parity with main.py (registers ocp_irm)
    import irm_trainer  # noqa: F401

    # Now load causal_model — registers V7 (and later V8) models + trainers
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "causal_model"))
    import causal_model.model_causal  # noqa: F401
    import causal_model.trainer_causal  # noqa: F401

    # Optional: also import equiformer_lora for cross-backbone support
    try:
        from fairchem.core.models.equiformer_v2 import equiformer_lora  # noqa: F401
    except Exception:
        pass

    # Patched build_config — forces correct trainer + propagates causal/irm
    # blocks into model config so trainers can read them at __init__.
    import fairchem.core.common.utils as _fu
    _orig_build_config = _fu.build_config

    def _patched_build_config(args, args_override, include_paths=None):
        config = _orig_build_config(args, args_override, include_paths)

        # IRM (legacy)
        irm_cfg = config.get("irm", {})
        if float(irm_cfg.get("lambda", 0.0)) > 0:
            config["trainer"] = "ocp_irm"
            config.setdefault("model", {})["irm"] = irm_cfg

        # Causal V7/V8/V9: force trainer + propagate config block.
        # If user yml has top-level `causal:`, base.yml's `trainer: ocp` is
        # overridden here regardless of include order.
        causal_cfg = config.get("causal", {})
        if causal_cfg:
            version = str(causal_cfg.get("version", "v7")).lower()
            if version == "v7":
                config["trainer"] = "ocp_causal_v7"
            elif version == "v8":
                config["trainer"] = "ocp_causal_v8"
            elif version == "v9":
                config["trainer"] = "ocp_causal_v9"
            else:
                raise ValueError(f"Unknown causal version: {version}")
            # Embed inside model dict for trainer to access via self.config["model"]["causal"]
            config.setdefault("model", {})["causal"] = causal_cfg
            # Keep top-level too for direct access via self.config["causal"]
            print(f"[causal_main] Forced trainer={config['trainer']}, "
                  f"stage={causal_cfg.get('stage', '?')}")
        return config

    _fu.build_config = _patched_build_config

    from fairchem.core._cli import main
    main()
