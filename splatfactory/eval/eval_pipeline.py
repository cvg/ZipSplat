"""Base evaluation pipeline: export -> eval loop, config persistence,
and result (de)serialization helpers shared by all benchmarks.

Author: Alexander Veicht
"""

import json

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from splatfactory import get_logger

logger = get_logger(__name__)


def load_eval(dir):
    summaries, results = {}, {}
    with h5py.File(str(dir / "results.h5"), "r") as hfile:
        for k in hfile.keys():
            r = np.array(hfile[k])
            if len(r.shape) < 3:
                results[k] = r
        for k, v in hfile.attrs.items():
            summaries[k] = v
    with open(dir / "summaries.json", "r") as f:
        s = json.load(f)
    summaries = {k: v if v is not None else np.nan for k, v in s.items()}
    return summaries, results


def save_eval(dir, summaries, figures, results):
    with h5py.File(str(dir / "results.h5"), "w") as hfile:
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(arr.dtype, np.number):
                arr = arr.astype("object")
            hfile.create_dataset(k, data=arr)
        # just to be safe, not used in practice
        for k, v in summaries.items():
            hfile.attrs[k] = v
    s = {
        k: float(v) if np.isfinite(v) else None
        for k, v in summaries.items()
        if not isinstance(v, list)
    }
    s = {**s, **{k: v for k, v in summaries.items() if isinstance(v, list)}}
    with open(dir / "summaries.json", "w") as f:
        json.dump(s, f, indent=4)

    for fig_name, fig in figures.items():
        fig.savefig(dir / f"{fig_name}.png")


def exists_eval(dir):
    return (dir / "results.h5").exists() and (dir / "summaries.json").exists()


class EvalPipeline:
    default_conf = {
        "max_samples": None,  # Number of samples to run eval on (None=all)
    }
    eval_data_conf = {}

    export_keys = ()
    optional_export_keys = ()

    default_x: str | None = None  # Default x-axis for inspection plots
    default_y: str | None = None  # Default y-axis for inspection plots

    max_samples: int | None = None  # Number of samples to run eval on (None=all)

    main_metric = "???"  # You need to define this.

    def __init__(self, conf):
        """Assumes"""
        self.default_conf = OmegaConf.create(self.default_conf)
        self.conf = OmegaConf.merge(EvalPipeline.default_conf, self.default_conf, conf)
        self.__class__.max_samples = self.conf.max_samples
        self.export_keys = list(self.export_keys)
        self.optional_export_keys = list(self.optional_export_keys)
        self._init(self.conf)

    def _init(self, conf):
        pass

    @classmethod
    def get_dataset(self, data_conf=None):
        """Returns a dataset with samples for each eval datapoint"""
        return self.get_dataloader(data_conf).dataset

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        raise NotImplementedError

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        raise NotImplementedError

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        raise NotImplementedError

    def run(self, experiment_dir, model=None, overwrite=False, overwrite_eval=False):
        """Run export+eval loop"""
        self.save_conf(experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval)
        logger.info(f"Running eval pipeline {self.__class__.__name__}.")
        logger.info(f'Loop 1: Exporting predictions to "{experiment_dir}".')
        with torch.no_grad():
            pred_file = self.get_predictions(experiment_dir, model=model, overwrite=overwrite)
        logger.info(f"Loop 1 finished. Predictions saved to {pred_file}.")

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            logger.info(f"Loop 2: Evaluating predictions in {pred_file}.")
            s, f, r = self.run_eval(
                self.get_dataloader(
                    OmegaConf.merge(self.default_conf["data"], self.eval_data_conf)
                ),
                pred_file,
            )
            save_eval(experiment_dir, s, f, r)
            logger.info(f"Loop 2 finished. Results saved to {experiment_dir}.")
        s, r = load_eval(experiment_dir)
        return s, f, r

    def save_conf(self, experiment_dir, overwrite=False, overwrite_eval=False):
        conf_output_path = experiment_dir / "conf.yaml"
        if conf_output_path.exists() and not overwrite:
            saved_conf = OmegaConf.load(conf_output_path)
            assert (
                saved_conf.data == self.conf.data and saved_conf.model == self.conf.model
            ), "configs changed, add --overwrite to rerun experiment with new conf"
            assert (
                overwrite_eval or saved_conf.eval == self.conf.eval
            ), "eval configs changed, add --overwrite_eval to rerun evaluation"
        OmegaConf.save(self.conf, conf_output_path)
