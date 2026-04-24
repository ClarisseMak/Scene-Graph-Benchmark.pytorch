from contextlib import contextmanager


try:
    from apex import amp as amp  # type: ignore
except Exception:
    class _AmpCompat(object):
        """Minimal apex.amp compatibility layer for environments without NVIDIA apex."""

        @staticmethod
        def float_function(func):
            return func

        @staticmethod
        def init(enabled=True, verbose=False, *args, **kwargs):
            return None

        @staticmethod
        def initialize(model, optimizer=None, opt_level="O0", *args, **kwargs):
            if optimizer is None:
                return model
            return model, optimizer

        @staticmethod
        @contextmanager
        def scale_loss(loss, optimizer, *args, **kwargs):
            yield loss

    amp = _AmpCompat()
