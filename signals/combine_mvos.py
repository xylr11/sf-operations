import argparse
import polars as pl
from pathlib import Path
import sys

def combine_weights(weights_dir: Path, out_file: Path | None, lazy: bool=False) -> None:
    """
    Combine weights from folder
    Parameters
    ----------
    weights_dir : Path
        Folder with weights from get_signal_weights
    out_file : Path
        Path to write output parquet
    lazy : bool
        Optionally use lazy scanning for large jobs

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing optimized portfolio weights across
        all backtest dates, with columns:

        - ``date`` : datetime, portfolio date
        - ``barrid`` : str, asset identifier
        - Columns of weights
    """
    # Collect all .parquet files (non-recursive)
    files = sorted(weights_dir.glob("*.parquet"))
    if not files:
        sys.exit(f"[ERROR] No parquet files found in {weights_dir.resolve()}")

    dfs = []
    for f in files:
        signal = f.stem.split("_")[0]  # e.g. momentum_2024.parquet → momentum
        reader = pl.scan_parquet if lazy else pl.read_parquet
        df = (
            reader(f)
            .select(["date", "barrid", "weight"])
            .with_columns(pl.lit(signal).alias("signal"))
        )
        dfs.append(df)

    # Combine and pivot
    combined = pl.concat(dfs)
    if lazy:
        combined = combined.collect()  # Collect after transformations

    combined = (
        combined.pivot(
            index=["date", "barrid"],
            columns="signal",
            values="weight",
            aggregate_function=None,
        )
        .sort(["date", "barrid"])
    )

    # Rename columns to include "_weight"
    weight_cols = [c for c in combined.columns if c not in ("date", "barrid")]
    combined = combined.rename({c: f"{c}_weight" for c in weight_cols})

    # Determine output path
    if out_file is None:
        out_file = weights_dir / "updated_weights_pivot.parquet"

    combined.write_parquet(out_file)

    print(f"[INFO] Combined dataframe written to {out_file}")
    print(combined.head())

    return combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine and pivot signal weight parquet files"
    )
    parser.add_argument(
        "--weights_dir",
        type=Path,
        default=Path("weights"),
        help="Directory containing parquet files (default: ./weights)",
    )
    parser.add_argument(
        "--out_file",
        type=Path,
        default=None,
        help="Optional output parquet path (default: weights/updated_weights_pivot.parquet)",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        action="store_true",
        help="Use Polars lazy scanning for efficiency on large datasets",
    )
    args = parser.parse_args()

    combine_weights(args.weights_dir, args.out_file, args.lazy)
