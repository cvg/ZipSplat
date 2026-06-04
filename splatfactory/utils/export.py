"""Export model predictions to h5.

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from splatfactory.utils import mappings


@torch.no_grad()
def export_predictions(
    loader,
    model,
    output_file,
    keys="*",
    optional_keys=[],
    callback_fn=None,
    mode="w",
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    hfile = h5py.File(str(output_file), mode)
    for data_ in tqdm(loader, desc="Export", ncols=100):
        data = mappings.batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}

        # Filter keys
        all_keys = set(pred.keys())
        if keys != "*":
            matched = []
            for pattern in keys:
                found = [k for k in all_keys - set(matched) if pattern in k]
                assert found, f"Pattern {pattern} not found in prediction keys."
                matched.extend(found)
        else:
            matched = list(all_keys)
        for pattern in optional_keys:
            matched.extend(k for k in all_keys - set(matched) if pattern in k)

        pred = {k: pred[k] for k in matched}
        pred = mappings.remove_batch_dim(pred)
        pred = mappings.batch_to_numpy(pred)

        try:
            grp = hfile.create_group(data["name"][0])
            dict_to_h5group(grp, pred)
        except RuntimeError:
            print(f"Skipping {data['name'][0]} (already in file?)")

        del pred, data, data_
        torch.cuda.empty_cache()

    hfile.close()
    return output_file


def dict_to_h5group(h5grp, data):
    """Write a nested dictionary to an h5 file."""
    for k, v in data.items():
        if isinstance(v, dict):
            dict_to_h5group(h5grp.create_group(k), v)
        elif isinstance(v, np.ndarray):
            h5grp.create_dataset(k, data=v)
        elif isinstance(v, torch.Tensor):
            h5grp.create_dataset(k, data=v.cpu().numpy())
        else:
            h5grp.attrs[k] = v


def dict_to_h5(file_path, data, modfe="w"):
    """Write a nested dictionary to an h5 file."""
    with h5py.File(file_path, modfe) as hfile:
        dict_to_h5group(hfile, data)
    return file_path
