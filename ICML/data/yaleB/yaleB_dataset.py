from typing import Union
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn import functional as F


class YaleBDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform=None, dataframe=None):
        if dataframe is not None:
            self.data = dataframe.reset_index(drop=True)
        else:
            self.data = pd.read_csv(csv_path)

        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['file'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        pose_category = row['pose_category']

        # min max normalization for azimuth and elevation
        azimuth = row['azimuth']
        elevation = row['elevation']
        azimuth_norm = (azimuth + 130) / 260  # Normalize to [0, 1]
        elevation_norm = (elevation + 35) / 100  # Normalize to [0, 1]

        partition_onehot = F.one_hot(torch.tensor(row['partition']), num_classes=3).float()
        partition = torch.tensor(row['partition'], dtype=torch.float32)

        elevation_azimuth = torch.tensor([azimuth_norm, elevation_norm], dtype=torch.float32)

        ps_max = 2.61
        ps_min = -1.82
        # projection_score = (row['projection_score'] - ps_min) / (ps_max - ps_min)
        projection_score = row['projection_score']

        projection_score = torch.tensor(projection_score, dtype=torch.float32)

        return {
            'img': image,
            'label_cat': torch.tensor(pose_category, dtype=torch.long),
            # 'label_cat': torch.tensor(row["partition"], dtype=torch.long),
            'azimuth': torch.tensor(azimuth_norm, dtype=torch.float32),
            'elevation': torch.tensor(elevation_norm, dtype=torch.float32),
            'b': elevation_azimuth,
            }


class YaleBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        train_csv,
        val_csv,
        test_csv,
        batch_size=64,
        num_workers=4,
        img_size=224,
        circe_enabled=False,
        circe_heldout_size: Union[int, float] = 0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.circe_enabled = circe_enabled
        self.circe_heldout_size = circe_heldout_size

        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.1444, 0.1444, 0.1444], std=[0.2104, 0.2104, 0.2104])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.1444, 0.1444, 0.1444], std=[0.2104, 0.2104, 0.2104])
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_train_df = pd.read_csv(self.train_csv)

            if self.circe_enabled:
                train_df, heldout_df = train_test_split(
                    full_train_df,
                    test_size=self.circe_heldout_size if isinstance(self.circe_heldout_size, float) else None,
                    train_size=None if isinstance(self.circe_heldout_size, float) else len(full_train_df) - self.circe_heldout_size,
                    random_state=42,
                    shuffle=True,
                )
                self.train_dataset = YaleBDataset(
                    csv_path=None,
                    root_dir=self.data_dir,
                    transform=self.train_transform,
                    dataframe=train_df,
                )
                self.heldout_dataset_circe = YaleBDataset(
                    csv_path=None,
                    root_dir=self.data_dir,
                    transform=self.test_transform,
                    dataframe=heldout_df,
                )
            else:
                self.train_dataset = YaleBDataset(
                    csv_path=self.train_csv,
                    root_dir=self.data_dir,
                    transform=self.train_transform,
                )

            self.val_dataset = YaleBDataset(
                csv_path=self.val_csv,
                root_dir=self.data_dir,
                transform=self.test_transform,
            )

        if stage == 'test' or stage is None:
            self.test_dataset = YaleBDataset(
                csv_path=self.test_csv,
                root_dir=self.data_dir,
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def heldout_dataloader(self):
        if not self.circe_enabled:
            raise ValueError("CIRCE is not enabled.")
        return DataLoader(
            self.heldout_dataset_circe,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    

if __name__ == "__main__":
    dataset = YaleBDataset(
        csv_path="/datasets/yaleB/ExtendedYaleB/_base_splits/train_partitioned_biased.csv",
        root_dir="/datasets/yaleB/ExtendedYaleB",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    mean = 0.0
    std = 0.0
    total_images_count = 0

    for batch in loader:
        images = batch['img']  # [B, C, H, W]
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)  # [B, C, H*W]
        mean += images.mean(dim=2).sum(dim=0)
        std += images.std(dim=2).sum(dim=0)
        total_images_count += batch_size

    mean /= total_images_count
    std /= total_images_count
    print(f"Mean: {mean}, Std: {std}")
    print("Dataset loaded and processed successfully.")
    print(f"Total images: {len(dataset)}")
