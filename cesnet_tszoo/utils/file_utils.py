import pickle
from typing import Any

import numpy as np

import yaml


def pickle_load(path: str):
    # Help NumPy 1.x unpickle NumPy 2.x pickles, please use this sparingly and only temporarily
    if np.__version__[:2] == "1.":
        import sys
        sys.modules["numpy._core.numeric"] = np.core.numeric

    with open(path, "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object


def pickle_dump(object_to_save: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(object_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)


def yaml_load(path: str) -> Any:
    with open(path, "r") as f:
        loaded_object = yaml.load(f, Loader=yaml.SafeLoader)
    return loaded_object


def yaml_dump(object_to_save: Any, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(object_to_save, f)
