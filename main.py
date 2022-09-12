from typing import Any
import glob
import os

import prior
from tqdm import tqdm


def load_dataset(config: Any = None) -> prior.DatasetDict:
    data = dict(
        train=prior.Dataset(data=[], dataset="rearrangement_episodes", split="train"),
        val=prior.Dataset(data=[], dataset="rearrangement_episodes", split="val"),
    )

    files = glob.glob(os.path.join("2021", f"*.pkl.gz"))

    all_splits = []
    for file in tqdm(files, total=len(files), desc=f"Loading 2021ithor"):
        with open(file, "rb") as f:
            bytes = f.read(-1)
        all_splits.append((os.path.basename(file).replace(".pkl.gz", ""), bytes))

    data["test"] = prior.Dataset(
        data=all_splits, dataset="rearrangement_episodes", split="test",
    )

    return prior.DatasetDict(**data)
