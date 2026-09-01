import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


class MNLIDataset(Dataset):
    def __init__(self, 
                 csv_path: str = None, 
                 tensor_path: str = None, 
                 dataframe: pd.DataFrame = None, 
                 tensors: dict = None, 
                 split_indices: list = None):
        
        # ... (Loading logic matches previous file exactly) ...
        if dataframe is not None:
            self.data = dataframe.reset_index(drop=True)
        else:
            self.data = pd.read_csv(csv_path)

        if tensors is not None:
            self.tensors = tensors
        else:
            self.tensors = torch.load(tensor_path)
            
        self.split_indices = split_indices

    def __len__(self):
        if self.split_indices is not None:
            return len(self.split_indices)
        return len(self.data)

    def __getitem__(self, idx):
        if self.split_indices is not None:
            physical_idx = self.split_indices[idx]
        else:
            physical_idx = idx

        # 1. Get Text Features
        input_ids = self.tensors['input_ids'][physical_idx]
        attention_mask = self.tensors['attention_mask'][physical_idx]
        token_type_ids = self.tensors['token_type_ids'][physical_idx]

        # 2. Get Metadata
        row = self.data.iloc[idx]
        
        label = torch.tensor(row['gold_label'], dtype=torch.long)
        negation = torch.tensor(row['sentence2_has_negation'], dtype=torch.float32)

        # 3. CONSTRUCT BATCH
        # We pack the BERT inputs into "img" so the LightningModule accepts them blindly.
        return {
            "img": {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            },
            "label_cat": label,
            "b": negation,
            "pairID": str(row['pairID'])
        }


class MNLIDataModule(pl.LightningDataModule):
    def __init__(self, csv_dir, tensor_dir, batch_size=32, num_workers=4, circe_enabled=False, circe_heldout_size=0.2, seed=42):
        super().__init__()
        self.csv_dir = csv_dir
        self.tensor_dir = tensor_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.circe_enabled = circe_enabled
        self.circe_heldout_size = circe_heldout_size
        self.seed = seed

    def setup(self, stage=None):
        if stage in ('fit', None):
            train_csv = os.path.join(self.csv_dir, 'train.csv')
            train_pt = os.path.join(self.tensor_dir, 'train_features.pt')
            full_train_df = pd.read_csv(train_csv)
            full_train_tensors = torch.load(train_pt)

            if self.circe_enabled:
                indices = list(range(len(full_train_df)))
                train_idx, heldout_idx = train_test_split(indices, test_size=self.circe_heldout_size, random_state=self.seed, shuffle=True)
                
                self.train_dataset = MNLIDataset(dataframe=full_train_df.iloc[train_idx].reset_index(drop=True), tensors=full_train_tensors, split_indices=train_idx)
                self.heldout_dataset_circe = MNLIDataset(dataframe=full_train_df.iloc[heldout_idx].reset_index(drop=True), tensors=full_train_tensors, split_indices=heldout_idx)
            else:
                self.train_dataset = MNLIDataset(dataframe=full_train_df, tensors=full_train_tensors)

            self.val_dataset = MNLIDataset(csv_path=os.path.join(self.csv_dir, 'val.csv'), tensor_path=os.path.join(self.tensor_dir, 'val_features.pt'))

        if stage in ('test', None):
            self.test_dataset = MNLIDataset(csv_path=os.path.join(self.csv_dir, 'test.csv'), tensor_path=os.path.join(self.tensor_dir, 'test_features.pt'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    def heldout_dataloader(self):
        return DataLoader(self.heldout_dataset_circe, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)