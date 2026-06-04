from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = root / "data/"  # datasets
CHECKPOINT_PATH = root / "weights/"  # pretrained weights path
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
EVAL_PATH = root / "outputs/results/"  # evaluation results
THIRD_PARTY_PATH = root / "third_party/"  # third-party libraries

ALLOW_PICKLE = True  # allow pickle (e.g. in torch.load)
