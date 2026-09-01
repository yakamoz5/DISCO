from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import torch
import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms


class DspritesDataset(Dataset):
    def __init__(self, h5_path, csv_path=None, dataframe=None, transform=None):
        self.h5_path = h5_path
        self.df = dataframe.reset_index(drop=True) if dataframe is not None else pd.read_csv(csv_path)
        self.transform = transform

        with h5py.File(h5_path, 'r') as f:
            self.imgs = f["images"][:]  # Assuming images are in 'images' dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_index = int(row['index'])
        img = self.imgs[img_index]  # [64, 64], float32

        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        if self.transform:
            img = self.transform(img)

        # Convert all tabular fields individually
        # Continuous values (normalized in generation)
        x = torch.tensor(row['x'], dtype=torch.float32)
        y = torch.tensor(row['y'], dtype=torch.float32)
        scale = torch.tensor(row['scale'], dtype=torch.float32)
        orientation = torch.tensor(row['orientation'], dtype=torch.float32)

        x_im = torch.tensor(row['x_im'], dtype=torch.float32)
        y_im = torch.tensor(row['y_im'], dtype=torch.float32)
        scale_im = torch.tensor(row['scale_im'], dtype=torch.float32)
        orientation_im = torch.tensor(row['orientation_im'], dtype=torch.float32)

        # Shape categories
        shape_map = {'square': 0, 'ellipse': 1, 'heart': 2}
        shape_idx = shape_map[row['shape']]
        shape_one_hot = torch.nn.functional.one_hot(torch.tensor(shape_idx), num_classes=3).float()

        shape_im_idx = shape_map[row['shape_im']]
        shape_im_one_hot = torch.nn.functional.one_hot(torch.tensor(shape_im_idx), num_classes=3).float()

        # triple the channels
        img = img.repeat(3, 1, 1)  # [3, 64, 64]

        return {
            "img": img,                              # [3, 64, 64] or [1, 64, 64] if grayscale
            "index": torch.tensor(img_index),        # scalar int64

            # Original tabular values
            "x": x,
            "label_c": y,
            "scale": scale,
            "orientation": orientation,
            "shape": shape_one_hot,                  # [3]

            # Modified image tabular values
            "x_im": x_im,
            "y_im": y_im,
            "scale_im": scale_im,
            "orientation_im": orientation_im,
            "shape_im": shape_im_one_hot,            # [3]
        }


class DspritesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path,
        train_csv,
        val_csv,
        test_csv,
        batch_size=64,
        num_workers=4,
        circe_enabled=False,
        circe_heldout_size=0.2,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.circe_enabled = circe_enabled
        self.circe_heldout_size = circe_heldout_size

        # Normalization for grayscale images
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.1387], std=[0.3096])  # or set after computing dataset stats
        ])

    def setup(self, stage=None):
        if stage in ('fit', None):
            full_train_df = pd.read_csv(self.train_csv)

            if self.circe_enabled:
                from sklearn.model_selection import train_test_split
                train_df, heldout_df = train_test_split(
                    full_train_df,
                    test_size=self.circe_heldout_size if isinstance(self.circe_heldout_size, float) else None,
                    train_size=None if isinstance(self.circe_heldout_size, float) else len(full_train_df) - self.circe_heldout_size,
                    random_state=42,
                    shuffle=True,
                )
                self.train_dataset = DspritesDataset(
                    h5_path=self.h5_path,
                    dataframe=train_df,
                    transform=self.transform
                )
                self.heldout_dataset_circe = DspritesDataset(
                    h5_path=self.h5_path,
                    dataframe=heldout_df,
                    transform=self.transform
                )
            else:
                self.train_dataset = DspritesDataset(
                    h5_path=self.h5_path,
                    csv_path=self.train_csv,
                    transform=self.transform
                )

            self.val_dataset = DspritesDataset(
                h5_path=self.h5_path,
                csv_path=self.val_csv,
                transform=self.transform
            )

        if stage in ('test', None):
            self.test_dataset = DspritesDataset(
                h5_path=self.h5_path,
                csv_path=self.test_csv,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,  # Ensure consistent batch size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def heldout_dataloader(self):
        if not self.circe_enabled:
            raise ValueError("CIRCE is not enabled.")
        return DataLoader(
            self.heldout_dataset_circe,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":
    dataset = DspritesDataset(
        h5_path="/datasets/dSprites/sprites.h5",
        csv_path="/datasets/dSprites/train.csv"
        )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    mean = 0.0
    std = 0.0
    total = 0

    for batch in loader:
        imgs = batch["img"]  # [B, 1, 64, 64]
        B = imgs.size(0)
        imgs = imgs.view(B, -1)
        mean += imgs.mean(dim=1).sum()
        std += imgs.std(dim=1).sum()
        total += B

    mean /= total
    std /= total
    print(f"Mean: {mean}, Std: {std}")
