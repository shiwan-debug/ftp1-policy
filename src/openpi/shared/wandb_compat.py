import logging
import os
from typing import Any


def _read_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


# Macro switch:
# - True  -> use swanlab backend
# - False -> use wandb backend
# Default is True as requested.
USE_SWANLAB = _read_env_bool("USE_SWANLAB", True)

_backend_import_error: Exception | None = None

if USE_SWANLAB:
    try:
        import swanlab as _backend

        BACKEND_NAME = "swanlab"
    except Exception as exc:
        _backend_import_error = exc
        import wandb as _backend

        BACKEND_NAME = "wandb"
        USE_SWANLAB = False
else:
    import wandb as _backend

    BACKEND_NAME = "wandb"


class _NoopConfig:
    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _NoopRun:
    id = "disabled"
    url = ""
    config = _NoopConfig()

    def log_code(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_NOOP_RUN = _NoopRun()


class _RunProxy:
    def __init__(self, run: Any | None) -> None:
        self._run = run

    def __getattr__(self, name: str):
        if self._run is None:
            if name == "id":
                return _NOOP_RUN.id
            if name == "url":
                return _NOOP_RUN.url
            if name == "config":
                return _NOOP_RUN.config
            if name == "log_code":
                return _NOOP_RUN.log_code
            raise AttributeError(name)
        if hasattr(self._run, name):
            return getattr(self._run, name)
        if name == "id":
            return "run"
        if name == "url":
            return ""
        if name == "config":
            return _NOOP_RUN.config
        if name == "log_code":
            return _NOOP_RUN.log_code
        raise AttributeError(name)


class _ExperimentLoggerProxy:
    def __init__(self) -> None:
        self._disabled = False
        self._run: Any = None
        if _backend_import_error is not None:
            logging.warning(
                "USE_SWANLAB=True but swanlab import failed (%r). Falling back to wandb.",
                _backend_import_error,
            )

    @property
    def Image(self):
        image_ctor = getattr(_backend, "Image", None)
        if image_ctor is not None:
            return image_ctor
        return lambda data, *args, **kwargs: data

    @property
    def run(self):
        if self._disabled:
            return _NOOP_RUN
        backend_run = getattr(_backend, "run", None)
        if backend_run is not None:
            return _RunProxy(backend_run)
        if self._run is not None:
            return _RunProxy(self._run)
        return _NOOP_RUN

    @property
    def config(self):
        if self._disabled:
            return _NOOP_RUN.config
        backend_cfg = getattr(_backend, "config", None)
        if backend_cfg is not None:
            return backend_cfg
        run_cfg = getattr(self.run, "config", None)
        if run_cfg is not None:
            return run_cfg
        return _NOOP_RUN.config

    def init(self, *args: Any, **kwargs: Any):
        mode = kwargs.get("mode")
        if mode == "disabled":
            self._disabled = True
            self._run = _NOOP_RUN
            return _NOOP_RUN

        self._disabled = False
        init_kwargs = dict(kwargs)
        if BACKEND_NAME == "swanlab":
            # Keep args compatible when old wandb-specific kwargs are present.
            init_kwargs.pop("entity", None)
            init_kwargs.pop("sync_tensorboard", None)
            init_kwargs.pop("monitor_gym", None)
            init_kwargs.pop("save_code", None)
            init_kwargs.pop("mode", None)
            init_kwargs.pop("id", None)
            init_kwargs.pop("resume", None)

        self._run = _backend.init(*args, **init_kwargs)
        return self._run

    def log(self, data: dict[str, Any], step: int | None = None):
        if self._disabled:
            return None

        if step is None:
            return _backend.log(data)

        try:
            return _backend.log(data, step=step)
        except TypeError:
            payload = dict(data)
            payload.setdefault("step", step)
            return _backend.log(payload)

    def finish(self):
        if self._disabled:
            return None
        finish_fn = getattr(_backend, "finish", None)
        if callable(finish_fn):
            return finish_fn()
        return None

    def __getattr__(self, name: str):
        return getattr(_backend, name)


wandb = _ExperimentLoggerProxy()
