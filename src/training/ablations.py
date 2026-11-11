"""M3 ablation commands. Training is started only by scripts/run_ablations.py."""

import sys

CONFIG = "configs/proposed/e7_ft_rcl_atadd_v1.yaml"
SEED = 42
SEED_TAG = "seed1"

# Non-default M3 settings. The untouched config remains the full-gate MSE run.
ABLATION_VARIANTS = (
    ("gate_no_confidence", ("ft_rcl.gate=no_confidence",)),
    ("gate_no_type", ("ft_rcl.gate=no_type",)),
    ("gate_none", ("ft_rcl.gate=none",)),
    ("pc_js", ("ft_rcl.pc_loss=js",)),
    ("gamma_ftsc_0", ("ft_rcl.gamma_ftsc=0.0",)),
    ("gamma_pc_0", ("ft_rcl.gamma_pc=0.0",)),
    ("beta_0", ("ft_rcl.beta=0.0",)),
    ("beta_0.25", ("ft_rcl.beta=0.25",)),
    ("beta_1", ("ft_rcl.beta=1.0",)),
)


def experiment_id(name):
    return f"E7_M3_ablation_{name}_{SEED_TAG}"


def build_ablation_command(name, overrides, python=None):
    command = [
        python or sys.executable,
        "-m",
        "src.training.train",
        "--config",
        CONFIG,
        "--mode",
        "M3",
        "--seed",
        str(SEED),
        "--experiment-id",
        experiment_id(name),
        "--eval-splits",
        "dev,eval",
    ]
    for item in overrides:
        command.extend(["--set", item])
    return command


def ablation_commands(python=None):
    return [build_ablation_command(name, overrides, python=python) for name, overrides in ABLATION_VARIANTS]
