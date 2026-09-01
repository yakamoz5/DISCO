# correlate_splits.py

import os
import pandas as pd
import numpy as np

def correlate_split(df, nl_type='mod', ood=False, seed=42):
    np.random.seed(seed)
    unique_poses = np.sort(df['pose'].unique())
    new_indices = []
    for pose in unique_poses:
        pose_idx = df.index[df['pose'] == pose].tolist()
        if nl_type == 'mod' and not ood:
            az_range = (df['azimuth'] >= 0.5 * np.abs(df['pose']/10) - 2) & \
                       (df['azimuth'] <= 0.5 * np.abs(df['pose']/10) + 2)
            az_idx = df.index[az_range].tolist()
            valid_idx = list(set(az_idx).intersection(pose_idx))
            new_indices.extend(valid_idx)
        elif nl_type == 'y-cone' and not ood:
            az = 0.5 * (df['azimuth'] / 130 + 1) * 9
            for _ in range(10):
                eps = 0.5 * 2 * (np.random.randint(0, 2, size=1) - 0.5) * pose ** 2 / 9
                z_idx = df.index[np.abs(az - 0.5 * pose - eps) < 1].tolist()
                valid_idx = list(set(z_idx).intersection(pose_idx))
                new_indices.extend(valid_idx)
        else:  # OOD: shuffle
            new_indices.extend(pose_idx)
    # Don't pad or truncate to len(df)!
    new_indices = np.unique(new_indices)  # avoid duplicates
    np.random.shuffle(new_indices)
    return df.iloc[new_indices].reset_index(drop=True)

def make_correlated_csvs(
    input_csv, output_dir, nl_type='mod', ood=False, n_trials=5, prefix='train'
):
    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_trials):
        correlated_df = correlate_split(df, nl_type=nl_type, ood=ood, seed=i)
        out_csv = os.path.join(output_dir, f"{prefix}_corr_{nl_type}_{'ood' if ood else 'id'}_{i}.csv")
        correlated_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--nl_type", type=str, default='y-cone', choices=['mod', 'y-cone'])
    parser.add_argument("--ood", action='store_true', default=False)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--prefix", type=str, default='train')
    args = parser.parse_args()
    make_correlated_csvs(
        args.csv, args.output_dir,
        nl_type=args.nl_type,
        ood=args.ood,
        n_trials=args.n_trials,
        prefix=args.prefix,
    )
