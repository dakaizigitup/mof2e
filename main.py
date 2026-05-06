"""For backwards compatibility"""

from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# 0:baseline_head_oldguiyi
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import irm_trainer  # noqa: F401 — 注册 ocp_irm trainer

    # Patch build_config so that configs with irm.lambda > 0:
    #   1. Force trainer = "ocp_irm"
    #   2. Embed irm config inside model config so it reaches IRMTrainer.__init__
    # Must happen BEFORE importing _cli (so _cli's `from utils import build_config` gets patched version)
    import fairchem.core.common.utils as _fu
    _orig_build_config = _fu.build_config

    def _patched_build_config(args, args_override, include_paths=None):
        config = _orig_build_config(args, args_override, include_paths)
        irm_cfg = config.get("irm", {})
        if float(irm_cfg.get("lambda", 0.0)) > 0:
            config["trainer"] = "ocp_irm"
            config["model"]["irm"] = irm_cfg
        return config

    _fu.build_config = _patched_build_config

    from fairchem.core._cli import main
    main()
