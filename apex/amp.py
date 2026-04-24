from contextlib import contextmanager


def float_function(func):
    """Compatibility shim for apex.amp.float_function."""
    return func


def init(enabled=True, verbose=False, *args, **kwargs):
    """Compatibility shim for apex.amp.init used in eval scripts."""
    return None


def initialize(model, optimizer=None, opt_level="O0", *args, **kwargs):
    """Compatibility shim for apex.amp.initialize used in train scripts."""
    if optimizer is None:
        return model
    return model, optimizer


@contextmanager
def scale_loss(loss, optimizer, *args, **kwargs):
    """Compatibility shim for apex.amp.scale_loss context manager."""
    yield loss
