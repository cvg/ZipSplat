"""CLI plumbing for benchmarks: argument parsing, config resolution
(incl. eval.num_views -> indices file), model loading, and the run_cli entry point.

Author: Alexander Veicht
"""

import argparse
from pathlib import Path
from pprint import pprint

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tabulate import tabulate

from splatfactory import get_logger, settings
from splatfactory.models import get_model
from splatfactory.utils import experiments

logger = get_logger(__name__)


def extract_benchmark_conf(conf, benchmark):
    mconf = OmegaConf.create({})
    for key in ("model", "data", "eval"):
        if key in conf:
            mconf[key] = conf[key]
    if "benchmarks" in conf.keys():
        return OmegaConf.merge(mconf, conf.benchmarks.get(benchmark, {}))
    else:
        return mconf


def parse_eval_args(benchmark, args, configs_path, default=None):
    conf = {"data": {}, "model": {}, "eval": {}}
    if args.conf:
        conf_path, custom_conf = experiments.compose_config(
            args.conf, default_config_dir=configs_path
        )
        conf = extract_benchmark_conf(OmegaConf.merge(conf, custom_conf), benchmark)
        args.tag = args.tag if args.tag is not None else conf_path.stem

    cli_conf = OmegaConf.from_cli(args.dotlist)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.checkpoint = args.checkpoint if args.checkpoint else conf.get("checkpoint")

    # Track if max_samples explicitly set before checkpoint merge
    explicit_max_samples = "max_samples" in conf or any(
        "max_samples" in arg for arg in args.dotlist
    )

    checkpoint_name = conf.checkpoint
    if conf.checkpoint and not conf.checkpoint.endswith(".tar"):
        if Path(conf.checkpoint).exists():
            checkpoint_dir = Path(conf.checkpoint).absolute()
            if checkpoint_dir.is_relative_to(settings.TRAINING_PATH):
                checkpoint_name = str(checkpoint_dir.relative_to(settings.TRAINING_PATH))
        else:
            checkpoint_dir = settings.TRAINING_PATH / conf.checkpoint
        checkpoint_conf = OmegaConf.load(checkpoint_dir / "config.yaml")
        # Only extract model config from checkpoint - data/eval come from
        # benchmark defaults, --conf file, and CLI overrides.
        model_conf = OmegaConf.create({"model": checkpoint_conf.get("model", {})})
        conf = OmegaConf.merge(model_conf, conf)

    # Default to full eval unless max_samples explicitly set
    if not explicit_max_samples and "max_samples" in conf:
        conf.max_samples = None

    if default:
        conf = OmegaConf.merge(default, conf)

    name = ""
    if args.tag:
        name = args.tag
    elif args.conf and checkpoint_name:
        name = f"{args.conf}_{checkpoint_name}"
    elif args.conf:
        name = args.conf
    elif checkpoint_name:
        name = checkpoint_name
    if len(args.dotlist) > 0 and not args.tag:
        name = name + "_" + ":".join(args.dotlist)

    if not name:
        raise ValueError("No tag provided. Please provide a tag with --tag or --conf.")
    logger.info("Running benchmark: %s", benchmark)
    logger.info("Experiment tag: %s", name)
    logger.info("Config:")
    pprint(OmegaConf.to_container(conf))
    return name, conf


def load_model(model_conf, checkpoint):
    if checkpoint:
        model = experiments.load_experiment(checkpoint, conf=model_conf).eval()
    else:
        model = get_model(model_conf.name)(model_conf).eval()
    if not model.is_initialized():
        raise ValueError(
            "The provided model has non-initialized parameters. "
            + "Try to load a checkpoint instead."
        )
    return model


def get_eval_parser(parser: argparse.ArgumentParser | None = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_eval", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    return parser


def run_cli(eval_cls, name: str, parser: argparse.ArgumentParser | None = None):
    """Run the evaluation pipeline from the command line."""
    parser = parser if parser is not None else get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(eval_cls.default_conf)

    # mingle paths
    output_dir = Path(settings.EVAL_PATH, name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(name, args, "configs/", default_conf)

    # `eval.num_views=N` selects the shipped eval index for this benchmark's view count.
    indices = getattr(eval_cls, "INDICES", None)
    num_views = OmegaConf.select(conf, "eval.num_views") if indices else None
    if num_views is not None:
        if num_views not in indices:
            raise ValueError(
                f"eval.num_views={num_views} not available for {name}; choose from {sorted(indices)}"
            )
        OmegaConf.update(conf, "data.view_sampler.indices_file", indices[num_views])

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True, parents=True)

    pipeline = eval_cls(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    print("\nSummary Dict:")
    pprint(s)

    table = tabulate([[k, v] for k, v in s.items()], headers=["Metric", "Value"], tablefmt="github")
    print("\nSummary:")
    print("\n" + table)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()

    return s, f, r
