"""Causal trainer module — auto-registers causal trainers.

  V7: ocp_causal_v7
  V8: ocp_causal_v8
  V9: ocp_causal_v9
"""
try:
    from . import causal_trainer_v7  # noqa: F401
except ImportError:
    pass
try:
    from . import causal_trainer_v8  # noqa: F401
except ImportError:
    pass
try:
    from . import causal_trainer_v9  # noqa: F401
except ImportError:
    pass
