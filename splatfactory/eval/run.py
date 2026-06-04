"""Run (multiple) benchmarks from the command line.

Author: Alexander Veicht
"""

# API: python -m splatfactory.eval.run benchmark1,benchmark2 --conf <name> ...

from splatfactory import get_logger

from . import get_benchmark, io

logger = get_logger(__name__)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmarks",
        type=lambda arg: arg.split(","),
        help="comma-separated list of benchmarks to run",
    )
    parser = io.get_eval_parser(parser)
    args = parser.parse_intermixed_args()
    logger.info(f"Running benchmarks {args.benchmarks}")
    for benchmark in args.benchmarks:
        benchmark_cls = get_benchmark(benchmark)
        io.run_cli(benchmark_cls, name=benchmark, parser=parser)
