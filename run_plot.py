import sys
from glob import glob
from pathlib import Path
from dataclasses import dataclass
import pyrallis
from src.utils import plot_stats


@dataclass
class PlotConfig:
    """Training config."""

    # Directory to save results
    results_dir: Path = Path("results")
    # Experiment name
    exp_name: Path = Path("default_exp")

    @property
    def exp_dir(self) -> Path:
        return self.results_dir / self.exp_name

def main(args=None) -> None:
    # If arguments were provided, replace sys.argv so Pyrallis sees them
    if args is not None and len(args) > 0:
        sys.argv = [sys.argv[0]] + args

    cfg = pyrallis.parse(config_class=PlotConfig)
    logfile = glob(f"{cfg.exp_dir}/*.jsonl")[0]
    plot_name = f"{cfg.exp_dir}/train_plot_{cfg.exp_name}.png"
    plot_stats(logfile, plot_name=plot_name)
    print(f"Plot saved to {plot_name}")


if __name__ == "__main__":
    main()