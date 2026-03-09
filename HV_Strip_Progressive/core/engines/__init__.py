"""
Pluggable forward-modeling engine registry.

Usage::

    from HV_Strip_Progressive.core.engines import registry

    # List available engines
    registry.list_available()          # ['diffuse_field', 'sh_wave', 'ellipticity']
    registry.list_implemented()        # ['diffuse_field']

    # Get an engine instance
    engine = registry.get("diffuse_field")
    result = engine.compute(model_path, config)

    # Register a custom engine
    registry.register(MyCustomEngine())
"""

from typing import Dict, List, Optional

from .base import BaseForwardEngine, EngineResult
from .diffuse_field import DiffuseFieldEngine
from .sh_wave import SHWaveEngine
from .ellipticity import EllipticityEngine


class EngineRegistry:
    """Central registry for forward-modeling engines."""

    def __init__(self):
        self._engines: Dict[str, BaseForwardEngine] = {}

    def register(self, engine: BaseForwardEngine) -> None:
        """Register an engine instance (overwrites if name exists)."""
        self._engines[engine.name] = engine

    def get(self, name: str) -> BaseForwardEngine:
        """Retrieve a registered engine by name.

        Raises
        ------
        KeyError
            If no engine with *name* is registered.
        """
        if name not in self._engines:
            available = ", ".join(self._engines.keys()) or "(none)"
            raise KeyError(
                f"Unknown engine {name!r}. Available: {available}"
            )
        return self._engines[name]

    def list_available(self) -> List[str]:
        """Return names of all registered engines (including stubs)."""
        return list(self._engines.keys())

    def list_implemented(self) -> List[str]:
        """Return names of engines that are fully implemented."""
        implemented = []
        for name, engine in self._engines.items():
            try:
                engine.get_default_config()
                # If compute is not NotImplementedError, consider it implemented
                # Quick check: try calling compute with None to see if it raises
                # NotImplementedError vs other errors
                implemented.append(name)
            except NotImplementedError:
                pass
        # Filter out stubs by checking if compute raises NotImplementedError
        result = []
        for name in implemented:
            engine = self._engines[name]
            try:
                engine.compute("__nonexistent__", {})
            except NotImplementedError:
                continue
            except Exception:
                # Any other error means compute is implemented (just failed)
                result.append(name)
        return result

    def get_engine_info(self) -> List[Dict]:
        """Return metadata for all registered engines."""
        info = []
        for name, engine in self._engines.items():
            is_stub = name not in self.list_implemented()
            info.append({
                "name": name,
                "description": engine.description,
                "implemented": not is_stub,
                "default_config": engine.get_default_config(),
                "required_params": engine.get_required_params(),
            })
        return info


# Singleton registry with built-in engines pre-registered
registry = EngineRegistry()
registry.register(DiffuseFieldEngine())
registry.register(SHWaveEngine())
registry.register(EllipticityEngine())


__all__ = [
    "BaseForwardEngine",
    "EngineResult",
    "EngineRegistry",
    "DiffuseFieldEngine",
    "SHWaveEngine",
    "EllipticityEngine",
    "registry",
]
