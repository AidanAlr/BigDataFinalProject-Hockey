#!/usr/bin/env python3
"""
Create a 20% subsample of the NHL data by sampling games.
Uses GameID to match events and results.

Usage:
  python create_subsample.py --events NHL_EventData.csv --results results.csv \
      --events-out NHL_EventData_subset.csv --results-out results_subset.csv
"""

import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Create a 20% subsample of NHL data by sampling games"
    )
    parser.add_argument(
        "--events",
        required=True,
        help="Path to input NHL_EventData.csv"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to input results.csv"
    )
    parser.add_argument(
        "--events-out",
        default="NHL_EventData_subset.csv",
        help="Path for output events subset (default: NHL_EventData_subset.csv)"
    )
    parser.add_argument(
        "--results-out",
        default="results_subset.csv",
        help="Path for output results subset (default: results_subset.csv)"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.20,
        help="Fraction of games to sample (default: 0.20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Read the results file first (smaller file)
    print(f"Reading {args.results}...")
    results_df = pd.read_csv(args.results)
    print(f"Results: {len(results_df)} rows")

    # Get unique game IDs from results
    unique_games = results_df['Game Id'].unique()
    print(f"Unique games in results: {len(unique_games)}")

    # Sample games
    sample_size = int(len(unique_games) * args.sample_rate)
    sampled_games = np.random.choice(unique_games, size=sample_size, replace=False)
    print(f"Sampled {len(sampled_games)} games ({args.sample_rate * 100:.0f}%)")

    # Filter results to sampled games
    results_subset = results_df[results_df['Game Id'].isin(sampled_games)]
    print(f"Results subset: {len(results_subset)} rows")

    # Save results subset
    results_subset.to_csv(args.results_out, index=False)
    print(f"Saved {args.results_out}")

    # Read events file in chunks due to large size
    print(f"\nReading {args.events} in chunks...")
    chunk_size = 500000
    events_chunks = []

    for i, chunk in enumerate(pd.read_csv(args.events, chunksize=chunk_size)):
        filtered_chunk = chunk[chunk['GameID'].isin(sampled_games)]
        events_chunks.append(filtered_chunk)
        print(f"Processed chunk {i+1}: kept {len(filtered_chunk)} rows")

    # Combine all chunks
    events_subset = pd.concat(events_chunks, ignore_index=True)
    print(f"\nEvents subset: {len(events_subset)} rows")

    # Save events subset
    events_subset.to_csv(args.events_out, index=False)
    print(f"Saved {args.events_out}")

    print("\nDone! Created:")
    print(f"  - {args.results_out} ({len(results_subset)} rows)")
    print(f"  - {args.events_out} ({len(events_subset)} rows)")


if __name__ == "__main__":
    main()
