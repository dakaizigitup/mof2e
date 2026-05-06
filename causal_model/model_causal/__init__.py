"""Causal model module — auto-registers V7/V8/V9 model classes.

Importing this package triggers registry.register_model for:
  V7: escn_v7_causal_s1 / s2 / s3
  V8: escn_v8_afcmg_s1 / s2 / s3
  V9: escn_v9_causal_s1 / s2 / s3

Existing model registrations (escn / escn_weighted_energy_head / equiformer_v2
/ equiformer_v2_lora) are NOT modified.
"""
from . import causal_modules    # noqa: F401  (imports + registers nothing on its own)
# Stage-specific model classes are registered lazily; uncomment when each is added.
try:
    from . import escn_v7_causal     # noqa: F401
except ImportError:
    pass
try:
    from . import escn_v8_afcmg     # noqa: F401
except ImportError:
    pass
try:
    from . import escn_v9_causal     # noqa: F401
except ImportError:
    pass
