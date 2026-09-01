# yaleB_csv_builder.py

import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

def parse_yaleb_filename(fn):
    """
    Example filename: 'yaleB01_P00A+000E+00.pgm'
    Returns: subject, pose, azimuth, elevation
    """
    base = os.path.basename(fn)
    # Assumes: yaleB[ID]_P[pose]A[+/-][azimuth]E[+/-][elevation].pgm
    subject = base.split('_')[0]  # yaleB01
    rest = base.split('_')[1]
    pose = int(rest[1:3])
    azimuth = int(rest[4:8])
    elevation = int(rest[9:12])

    return subject, pose, azimuth, elevation

def build_yaleb_csvs(root, outdir, val_size=0.1, test_size=0.1, random_state=42):
    """
    Build CSVs for train/val/test splits from raw YaleB folder.
    Each row: file, subject, pose, azimuth, elevation
    """
    image_paths = sorted(glob(os.path.join(root, "yaleB*/yaleB*_P*A*E*.pgm")))
    records = []
    for path in image_paths:
        subject, pose, az, el = parse_yaleb_filename(path)
        records.append({
            "file": os.path.relpath(path, root),
            "subject": subject,
            "pose": pose,
            "azimuth": az,
            "elevation": el,
        })
    df = pd.DataFrame(records)
    # Save full
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "yaleB_full.csv"), index=False)

    # Stratify split by subject (feel free to stratify by something else)
    subjects = df["subject"].unique()
    train_subj, test_subj = train_test_split(subjects, test_size=test_size, random_state=random_state)
    train_subj, val_subj = train_test_split(train_subj, test_size=val_size/(1-test_size), random_state=random_state)

    df_train = df[df.subject.isin(train_subj)].reset_index(drop=True)
    df_val = df[df.subject.isin(val_subj)].reset_index(drop=True)
    df_test = df[df.subject.isin(test_subj)].reset_index(drop=True)

    df_train.to_csv(os.path.join(outdir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(outdir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(outdir, "test.csv"), index=False)
    print(f"Wrote CSVs to {outdir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root directory of Extended YaleB (containing yaleB*)")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save CSVs")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_yaleb_csvs(args.root, args.outdir, val_size=args.val_size, test_size=args.test_size, random_state=args.seed)
