"""Loads cached model predictions from an HDF5 export, keyed by sample name, for evaluation.

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

import string

import h5py
import torch

from splatfactory import settings
from splatfactory.datasets import base_dataset
from splatfactory.models import BaseModel
from splatfactory.utils import mappings


def recursive_load(grp, pkeys):
    results = {}
    for k in pkeys:
        if k not in grp:
            # Key is an attribute
            val = grp.attrs[k]
        elif isinstance(grp[k], h5py.Dataset):
            # Key is a dataset - convert to tensor
            val = torch.from_numpy(grp[k].__array__())
        else:
            # Key is a group - recurse into it
            val = recursive_load(grp[k], list(grp[k].keys()))

        results[k] = val

    return results


class CacheLoader(BaseModel):
    default_conf = {
        "path": "???",  # can be a format string like exports/{scene}/
        "data_keys": None,  # load all keys
        "device": None,  # load to same device as data
        "trainable": False,
        "add_data_path": True,
        "collate": True,
        "padding_fn": None,
        "padding_length": None,  # required for batching!
        "numeric_type": "float32",  # [None, "float16", "float32", "float64"]
        "check_valid": True,  # check if points are inside the image after scaling
    }

    required_data_keys = ["name"]  # we need an identifier

    def _init(self, conf):
        self.hfiles = {}
        self.padding_fn = conf.padding_fn
        if self.padding_fn is not None:
            self.padding_fn = eval(self.padding_fn)
        self.numeric_dtype = {
            None: None,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[conf.numeric_type]

    def _forward(self, data):
        preds = []
        device = self.conf.device
        if not device:
            devices = set([v.device for v in data.values() if isinstance(v, torch.Tensor)])
            if len(devices) == 0:
                device = "cpu"
            else:
                assert len(devices) == 1
                device = devices.pop()

        var_names = [x[1] for x in string.Formatter().parse(self.conf.path) if x[1]]
        for i, name in enumerate(data["name"]):
            fpath = self.conf.path.format(**{k: data[k][i] for k in var_names})
            if self.conf.add_data_path:
                fpath = settings.DATA_PATH / fpath
            hfile = h5py.File(str(fpath), "r", locking=False)
            grp = hfile[name]
            pkeys = (
                self.conf.data_keys
                if self.conf.data_keys is not None
                else grp.keys() | grp.attrs.keys()
            )
            pred = recursive_load(grp, pkeys)
            if self.numeric_dtype is not None:
                pred = {
                    k: (
                        v
                        if not isinstance(v, torch.Tensor) or not torch.is_floating_point(v)
                        else v.to(dtype=self.numeric_dtype)
                    )
                    for k, v in pred.items()
                }
            pred = mappings.batch_to_device(pred, device)

            preds.append(pred)
            hfile.close()
        if self.conf.collate:
            return mappings.batch_to_device(base_dataset.collate(preds), device)
        else:
            assert len(preds) == 1
            return mappings.batch_to_device(preds[0], device)

    def loss(self, pred, data):
        raise NotImplementedError
