"""Test-Time Training: plastic adapters, Shampoo-lite, meta prior, continual trainer."""

from .adapter import PlasticAdapter as PlasticAdapter
from .adapter import TTTRouter as TTTRouter
from .meta_prior import MAMLTrainer as MAMLTrainer
from .shampoo_lite import ShampooLite as ShampooLite
from .trainer import TTTContinualTrainer as TTTContinualTrainer

__all__ = [
    "MAMLTrainer",
    "PlasticAdapter",
    "ShampooLite",
    "TTTContinualTrainer",
    "TTTRouter",
]
