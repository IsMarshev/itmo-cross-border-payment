"""A single public entry point: :class:`SignalEngine`."""

__version__ = "0.2.0"


def __getattr__(name):
    if name == "SignalEngine":
        from .engine import SignalEngine

        return SignalEngine
    raise AttributeError(name)
