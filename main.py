from typing import Any

import prior


def load_dataset(config: Any = None) -> prior.Dataset:
    """Load a dummy dataset."""
    return prior.Dataset(data=[], dataset="rearrangement_episodes", split="test")
