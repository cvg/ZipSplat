"""Runtime/environment helpers: device selection, dynamic class loading, threading, and RNG/seeding.

Author: Paul-Edouard Sarlin (skydes)
Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.

Timing/profiling helpers live in timing.py, optimizer/LR helpers in optim.py, and the
tensorboard summary writers in summary_writer.py.
"""

import inspect
import os
import random
from contextlib import contextmanager

import numpy as np
import torch

try:
    import mkl
except ImportError:  # optional: only used to cap MKL's thread count
    mkl = None


def get_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def get_class(mod_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """
    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def set_num_threads(nt):
    """Force numpy and other libraries to use a limited number of threads."""
    if mkl is not None:
        mkl.set_num_threads(nt)
    torch.set_num_threads(1)
    os.environ["IPC_ENABLE"] = "1"
    for o in [
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        os.environ[o] = str(nt)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_random_state(with_cuda):
    pth_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    if torch.cuda.is_available() and with_cuda:
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    return pth_state, np_state, py_state, cuda_state


def set_random_state(state):
    pth_state, np_state, py_state, cuda_state = state
    torch.set_rng_state(pth_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    if (
        cuda_state is not None
        and torch.cuda.is_available()
        and len(cuda_state) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def fork_rng(seed=None, with_cuda=True):
    state = get_random_state(with_cuda)
    if seed is not None:
        set_seed(seed)
    try:
        yield
    finally:
        set_random_state(state)
