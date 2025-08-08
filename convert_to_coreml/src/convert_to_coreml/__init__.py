# src/convert_to_coreml/__init__.py


from importlib.metadata import version, PackageNotFoundError
from .convert import main

__all__ = ["main", "__version__"]

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "1.0.0"
