from typing import Any
import glob
import os

import prior
from tqdm import tqdm


def load_dataset(config: Any = None) -> prior.DatasetDict:
    """Load the dataset."""
    data = {"test": []}

    for split in ["val", "train"]:
        directory_name = (
            f"split_{split}" if split not in ["val"] else f"split_mini_{split}"
        )
        files = sorted(
            glob.glob(
                os.path.join("2022procthor", directory_name, f"{split}_*_*.pkl.gz")
            ),
            key=lambda x: int(os.path.basename(x).split("_")[1]),
        )

        split_parts = []
        for file in tqdm(files, total=len(files), desc=f"Loading {split}"):
            with open(file, "rb") as f:
                bytes = f.read(-1)
            split_parts.append((os.path.basename(file).replace(".pkl.gz", ""), bytes))

        data[split] = prior.Dataset(
            data=split_parts,
            dataset="rearrangement_episodes",
            split=split,  # type:ignore
        )

    return prior.DatasetDict(**data)
