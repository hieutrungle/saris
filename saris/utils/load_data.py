import json
import numpy as np
from saris.utils import utils, buffers
import os


def load_offline_dataset(folder_dir: str, size: int) -> dict:
    data_names = [
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "truncations",
        "terminations",
    ]
    data = {}
    for i, name in enumerate(data_names):
        filename = os.path.join(folder_dir, f"{name}.txt")
        raw_data = _load_n_to_last_line(filename, size)
        first_entry = list(json.loads(raw_data[0]).values())[0]
        first_entry = np.asarray(first_entry)
        num_data_per_step = first_entry.shape[0]
        step_shape = first_entry.shape[1]
        preprocessed_data = np.full(
            shape=(len(raw_data) * num_data_per_step, step_shape), fill_value=np.nan
        )
        for i, step_data in enumerate(raw_data[::-1]):
            step_data: dict = json.loads(step_data)
            step_data = step_data.get(str(i), None)
            if step_data == None:
                raise f"Error in retrieve data at index {i} of data {name}"
            step_data = np.asarray(step_data)

            preprocessed_data[i : i + num_data_per_step] = step_data
        data[name] = preprocessed_data
    return data


def _load_n_to_last_line(filename: str, n: int = 1) -> list:

    container = []
    num_newlines = 0
    with open(filename, "rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b"\n":
                    num_newlines += 1
                    pos = f.tell()
                    container.append(f.readline().decode())
                    f.seek(pos)
        except OSError:
            f.seek(0)
            container.append(f.readline().decode())
    return container
