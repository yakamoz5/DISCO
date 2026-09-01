import json
import h5py
import numpy as np
from pathlib import Path
from pathlib import Path
import json
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Union


class SimDataset(Dataset):
    """
    A simple Dataset for simulated data saved in an HDF5 file.
    
    Each record in the HDF5 file is expected to be a group with the following datasets:
      - "img": a 2D NumPy array (H x W). This dataset will be unsqueezed to have a channel dimension.
      - "label": a scalar (0 or 1).
      - "cf": confounding effect value.
      - "mf": major effect value.
      
    The unique key (record id) is used as the identifier.
    """
    def __init__(self, h5_path: str, ids: list):
        """
        Args:
            h5_path: Path to the HDF5 file.
            ids: List of keys (as strings) to load from the file.
        """
        self.h5_path = h5_path
        self.ids = ids
        # Open the file in read-only mode.
        self.file = h5py.File(self.h5_path, "r")
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        key = self.ids[index]
        group = self.file[key]
        # Load image as a NumPy array and add a channel dimension.
        img = group["img"][:]  # shape: (H, W)
        img = np.expand_dims(img, axis=0)  # new shape: (1, H, W)
        img_tensor = torch.as_tensor(img, dtype=torch.float32)
        
        # Load other values.
        label = torch.as_tensor(group["label"][()], dtype=torch.float32)
        label_c = torch.as_tensor(group["label_c"][()], dtype=torch.float32)
        cf = torch.as_tensor(group["cf"][()], dtype=torch.float32)
        mf = torch.as_tensor(group["mf"][()], dtype=torch.float32)
        id = torch.as_tensor(int(key), dtype=torch.long)
        cf_std = torch.as_tensor(group["cf_std"][()], dtype=torch.float32)

        sample = {
            "id": id,
            "img": img_tensor,
            "label": label,
            "label_c": label_c,
            "cf": cf,
            "mf": mf,
            "cf_std": cf_std,
        }
        return sample

    def __del__(self):
        if hasattr(self, "file") and self.file:
            self.file.close()


class SimDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        split_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        circe_enabled: bool = False,
        circe_heldout_size: Union[int, float] = 0.2,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.circe_enabled = circe_enabled
        self.circe_heldout_size = circe_heldout_size

        self.h5 = self.data_dir

        split_file = split_dir
        with open(split_file, "r") as f:
            self.splits = json.load(f)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            if self.circe_enabled:
                train_ids, heldout_ids = train_test_split(
                    self.splits["train"],
                    test_size=self.circe_heldout_size if isinstance(self.circe_heldout_size, float) else None,
                    train_size=None if isinstance(self.circe_heldout_size, float) else len(self.splits["train"]) - self.circe_heldout_size,
                    random_state=42,
                    shuffle=True,
                )
                self.train_dataset = SimDataset(self.h5, train_ids)
                self.heldout_dataset_circe = SimDataset(self.h5, heldout_ids)
            else:
                self.train_dataset = SimDataset(self.h5, self.splits["train"])

            self.val_dataset = SimDataset(self.h5, self.splits["test"])

        if stage == "test" or stage is None:
            self.test_dataset = SimDataset(self.h5, self.splits["test"])

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
