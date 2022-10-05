import argparse
from pathlib import Path

from exp1 import main as main_1
from exp2 import main as main_2


def main():
    parser = argparse.ArgumentParser(description="Reproduce experiments from the paper")
    parser.add_argument("--n-repeats", type=int, help="Number of times to repeat each experiment", default=5)
    parser.add_argument("--max-iter", type=int, help="Max active learning steps allowed", default=150)
    parser.add_argument("--save-dir", type=Path, help="directory to save results", default=Path("tmp"))
    parser.add_argument("--n-jobs", type=int, help="Number of parallel jobs", default=15)

    args = parser.parse_args()

    main_1(
        misclassification_threshold=0.00,
        n_repeats=args.n_repeats,
        save_dir=args.save_dir,
        n_jobs=args.n_jobs,
        max_iter=args.max_iter,
    )
    main_2(
        misclassification_threshold=0.00,
        n_repeats=args.n_repeats,
        save_dir=args.save_dir,
        n_jobs=args.n_jobs,
        max_iter=args.max_iter,
    )
    main_1(
        misclassification_threshold=0.02,
        n_repeats=args.n_repeats,
        save_dir=args.save_dir,
        n_jobs=args.n_jobs,
        max_iter=args.max_iter,
    )
    main_2(
        misclassification_threshold=0.02,
        n_repeats=args.n_repeats,
        save_dir=args.save_dir,
        n_jobs=args.n_jobs,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
