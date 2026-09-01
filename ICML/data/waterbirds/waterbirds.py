import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split


class WaterbirdsDataset(Dataset):
    def __init__(self, csv_path: str = None, root_dir: str = None, dataframe: pd.DataFrame = None, train: bool = True, transform=None):
        """
        Args:
            csv_path (str): Path to train/val/test CSV file.
            root_dir (str): Root directory containing processed images.
            dataframe (pd.DataFrame): Preloaded DataFrame for circe split.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = dataframe.reset_index(drop=True) if dataframe is not None else pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.train = train

        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, scale=(0.8, 1.2), translate=(0.1, 0.1)),
                transforms.Normalize(mean=[0.4198, 0.4343, 0.3832], std=[0.2092, 0.2052, 0.2123]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4198, 0.4343, 0.3832], std=[0.2092, 0.2052, 0.2123]),
            ])

        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['unique_img_filename'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(row['y'], dtype=torch.float32)
        bias = torch.tensor(row['place'], dtype=torch.float32)

        return {
            'img': image,
            'label': label,
            'b': bias
        }


class WaterbirdsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        batch_size: int = 64,
        num_workers: int = 4,
        img_size: int = 224,
        circe_enabled: bool = False,
        circe_heldout_size: float = 0.2,
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

    def setup(self, stage=None):
        if stage in ('fit', None):
            full_train_df = pd.read_csv(self.train_csv)

            if self.circe_enabled:
                train_df, heldout_df = train_test_split(
                    full_train_df,
                    test_size=self.circe_heldout_size if isinstance(self.circe_heldout_size, float) else None,
                    train_size=None if isinstance(self.circe_heldout_size, float) else len(full_train_df) - self.circe_heldout_size,
                    random_state=42,
                    shuffle=True,
                )

                self.train_dataset = WaterbirdsDataset(
                    dataframe=train_df,
                    root_dir=self.data_dir,
                    train=True
                )

                self.heldout_dataset_circe = WaterbirdsDataset(
                    dataframe=heldout_df,
                    root_dir=self.data_dir,
                    train=False
                )
            else:
                self.train_dataset = WaterbirdsDataset(
                    csv_path=self.train_csv,
                    root_dir=self.data_dir,
                    train=True
                )

            self.val_dataset = WaterbirdsDataset(
                csv_path=self.val_csv,
                root_dir=self.data_dir,
                train=False
            )

        if stage in ('test', None):
            self.test_dataset = WaterbirdsDataset(
                csv_path=self.test_csv,
                root_dir=self.data_dir,
                train=False
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
    dataset = WaterbirdsDataset(
        csv_path="/datasets/waterbirds_bias/train.csv",
        root_dir="/datasets/waterbirds_bias/data",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for batch in loader:
        images = batch['img']
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(dim=2).sum(dim=0)
        std += images.std(dim=2).sum(dim=0)
        total_images_count += batch_size
    mean /= total_images_count
    std /= total_images_count
    print(f"Mean: {mean}, Std: {std}")
    print("Dataset loaded and processed successfully.")
    print(f"Total images: {len(dataset)}")
