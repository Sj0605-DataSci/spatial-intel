from .base import Agent
from .scripted import OracleAgent, NoisyAgent, RandomAgent

__all__ = ["Agent", "OracleAgent", "NoisyAgent", "RandomAgent", "make_flux"]


def make_flux(*args, **kwargs):
    """Lazy factory so importing this package never imports diffusers/torch-heavy bits."""
    from .flux import FluxAgent
    return FluxAgent(*args, **kwargs)
