import importlib
import sys
from jaxtyping import install_import_hook
from rich import print as rprint


def main():
    # Parse command-line args
    if len(sys.argv) < 2:
        rprint("Usage: uv run entry_point.py <command> [args...]")
        rprint("Commands:")
        rprint("- train: Run the training script")
        rprint("- plot: Run the plotting script")
        rprint("- eval: Run the evaluation script")
        rprint("- infer: Run the inference script")
        rprint("- benchmark: Run the benchmarking script")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    # Define the mapping of commands to scripts
    command_map = {
        "train": "run_train.py",
        "plot": "run_plot.py",
        "eval": "run_eval.py",
        "infer": "run_infer.py",
        "benchmark": "benchmark.py",
    }

    if command not in command_map:
        print(f"Unknown command: {command}")
        print("Available commands: train, plot, eval, infer, benchmark")
        sys.exit(1)

    with install_import_hook(["src"], "typeguard.typechecked"):
        if command == "train":
            run_train = importlib.import_module("run_train")
            run_train.main(args)
        elif command == "plot":
            run_plot = importlib.import_module("run_plot")
            run_plot.main(args)
        elif command == "eval":
            run_eval = importlib.import_module("run_eval")
            run_eval.main(args)
        elif command == "infer":
            run_infer = importlib.import_module("run_infer")
            run_infer.main(args)
        elif command == "benchmark":
            run_benchmark = importlib.import_module("scripts.benchmark")
            run_benchmark.main(args)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
