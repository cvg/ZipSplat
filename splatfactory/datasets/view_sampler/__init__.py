import importlib.util

from splatfactory.datasets.view_sampler.base_sampler import BaseViewSampler
from splatfactory.utils.tools import get_class


def get_view_sampler(name) -> BaseViewSampler:
    import_paths = [name, f"{__name__}.{name}"]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None

        if spec is not None:
            try:
                return get_class(path, BaseViewSampler)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_view_sampler__
                except AttributeError as exc:
                    print(exc)
                    continue

    raise RuntimeError(f'View sampler {name} not found in any of [{" ".join(import_paths)}]')
