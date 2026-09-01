import numpy as np
from sklearn.preprocessing import label_binarize
from ICML.resnet import TwoDResNet, ConvNet, Resnet18, Resnet50
from ICML.resnet_gn import TwoDResNetGN
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, AdamW, SGD, Adamax, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.cdisco_torch import sdisco_metric
from functools import partial

from omegaconf import DictConfig

from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score, mean_squared_error

import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TinyBERTPredictor(nn.Module):
    def __init__(self, model_name="huawei-noah/TinyBERT_General_4L_312D", **kwargs):
        super().__init__()
        # Load config to get hidden size
        config = AutoConfig.from_pretrained(model_name)
        self.feature_size = config.hidden_size  # Usually 312 for this model

        print(f"Loading TinyBERT model '{model_name}' with feature size {self.feature_size}...")
        
        # Load the model
        self.bert = AutoModel.from_pretrained(model_name)
        
    def forward(self, x):
        """
        Args:
            x: A dictionary containing 'input_ids', 'attention_mask', 'token_type_ids'
        """
        # We assume x is a dictionary coming from batch['img']
        outputs = self.bert(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )
        
        # HuggingFace models return (last_hidden_state, pooler_output)
        # However, TinyBERT might not have a trained pooler, or we might prefer the CLS token.
        # Let's extract the [CLS] token embedding (index 0) from the last hidden state.
        # Shape: [Batch_Size, Hidden_Size]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Your pipeline expects a tuple (logits, features). 
        # Your MetaDataPrediction class uses self.predictor(x)[1].
        # We return None for logits (as this is just the encoder) and features at index 1.
        return None, cls_embedding





class PredictionHead(torch.nn.Module):
    def __init__(self, in_features: int, n_outputs: int):
        super().__init__()
        # use layers with ReLU activation
        self.fc1 = torch.nn.Linear(in_features, 2*in_features)
        self.fc2 = torch.nn.Linear(2*in_features, in_features)
        self.fc3 = torch.nn.Linear(in_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PredictionHeadLinear(torch.nn.Module):
    def __init__(self, in_features: int, n_outputs: int):
        super().__init__()
        # use layers with ReLU activation
        self.fc = torch.nn.Linear(in_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MetaDataPredictionAbstract(pl.LightningModule):
    def __init__(
        self,
        resnet_cfg: dict,
        target: str,
        class_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        encoder_type: str = "resnet",
        pred_head: str = "linear",
        optimizer_cfg: Optional[DictConfig] = None,
        protected_attributes: Optional[List[str]] = None, 
    ):
        super().__init__()
        
        # Initialize Predictor
        if encoder_type == "resnet":
            from ICML.resnet import TwoDResNet # Late import to avoid breaking if file missing
            self.predictor = TwoDResNet(**resnet_cfg)
        elif encoder_type == "resnet_gn":
            from ICML.resnet_gn import TwoDResNetGN
            self.predictor = TwoDResNetGN(**resnet_cfg)
        elif encoder_type == "convnet":
            from ICML.resnet import ConvNet
            self.predictor = ConvNet(**resnet_cfg)
        elif encoder_type == "resnet_18":
            from ICML.resnet import Resnet18
            self.predictor = Resnet18(**resnet_cfg)
        elif encoder_type == "resnet_50":
            from ICML.resnet import Resnet50
            self.predictor = Resnet50(**resnet_cfg)
        elif encoder_type == "tinybert":
            # Assuming TinyBERTPredictor is in scope or imported
            self.predictor = TinyBERTPredictor(**resnet_cfg)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.optimizer_cfg = optimizer_cfg
        self.n_outputs = resnet_cfg.get("n_outputs", 1)

        prediction_head = PredictionHeadLinear if pred_head == "linear" else PredictionHead

        self.classifier_task = prediction_head(
            self.predictor.feature_size, 
            self.n_outputs
        )

        self.target = target
        self.automatic_optimization = False
        self.class_weights = (torch.as_tensor(class_weights, dtype=torch.float32)
                              if class_weights is not None else None)
        
        # Store protected attribute name for fetching 'b' from batch
        self.protected_attr_key = protected_attributes[0] if protected_attributes else "b"

        self.save_hyperparameters(ignore=["predictor"])

    def get_loss_function(self, target: str, weights: Optional[torch.Tensor] = None):
        if target in ["label"]:
            return BCEWithLogitsLoss(weight=weights)
        elif target in ["cf", "cf_std", "label_c"]:
            return F.mse_loss
        elif target in ["label_cat", "label_cat_ordered"]:
            return F.cross_entropy
        else:
            raise ValueError(f"Unsupported target type: {target}")

    def forward(self, x) -> torch.Tensor:
        return self.classifier_task(self.predictor(x)[1])

    def configure_optimizers(self):
        optimizer_map = {
            "adam": Adam, "adamw": AdamW, "sgd": SGD, "adamax": Adamax, "radam": RAdam
        }
        optimizer_cls = optimizer_map.get(self.optimizer_cfg.name)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_cfg.name}")

        optimizer_params = {k: v for k, v in self.optimizer_cfg.items() if k not in ["name", "lr_end"]}
        optimizer = optimizer_cls(
            list(self.predictor.parameters()) + list(self.classifier_task.parameters()), 
            **optimizer_params
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.optimizer_cfg.lr_end)
        return [optimizer], [scheduler]

    # --- Metric Calculation Logic ---

    def _compute_detailed_metrics(self, y: torch.Tensor, y_hat: torch.Tensor, b: torch.Tensor, target: str):
        """
        Computes standard metrics plus per-group accuracy and Worst Group Accuracy (WGA).
        """
        metrics = {}
        y_np = y.cpu().numpy()
        b_np = b.cpu().numpy()
        
        # Determine discrete predictions and integer labels
        if target == "label":
            y_pred = (y_hat > 0).float()
            y_long = y.long()
            y_pred_long = y_pred.long()
        elif target in ["label_cat", "label_cat_ordered"]:
            y_pred_long = torch.argmax(y_hat, dim=1)
            y_long = y.long()
        else:
            # Regression targets (no WGA logic applicable usually)
            return metrics

        # 1. Per Group Accuracy & WGA
        # Group is defined as intersection of Label and Bias (y, b)
        # Handle bias: if b is float (e.g. 0.0, 1.0), cast to long for grouping
        b_long = b.long()
        
        try:
            # Stack to find unique pairs (label, bias)
            # Shape [N, 2]
            groups = torch.stack((y_long, b_long), dim=1)
            unique_groups = torch.unique(groups, dim=0)
        except Exception as e:
            return metrics
        
        group_accuracies = []
        for g in unique_groups:
            g_lbl, g_bias = g[0].item(), g[1].item()
            # Create mask for this group
            mask = (y_long == g_lbl) & (b_long == g_bias)
            
            if mask.sum() > 0:
                acc = (y_pred_long[mask] == y_long[mask]).float().mean().item()
                group_accuracies.append(acc)
                # Store specific group accuracy
                metrics[f"acc_group_y{g_lbl}_b{g_bias}"] = acc
        
        if len(group_accuracies) > 0:
            metrics["wga"] = min(group_accuracies)
        else:
            metrics["wga"] = 0.0

        return metrics

    def calculate_log_metrics(self, y, y_hat, b, target, prefix="val"):
        y_np = y.cpu().numpy()
        y_hat_np = y_hat.cpu().numpy()

        # --- Standard Metrics ---
        if target == "label":
            y_pred = (y_hat_np > 0).astype(float)
            bacc = balanced_accuracy_score(y_np, y_pred)
            roc_auc = roc_auc_score(y_np, torch.sigmoid(y_hat).cpu().numpy())
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            self.log(f"{prefix}/label/bacc", bacc)
            self.log(f"{prefix}/label/roc_auc", roc_auc)

        elif target == "label_c":
            mse = F.mse_loss(y_hat, y)
            r2 = r2_score(y_np, y_hat_np)
            self.log(f"{prefix}/label_c/mse", mse)
            self.log(f"{prefix}/label_c/r2", r2)
            loss = mse

        elif target in ["label_cat", "label_cat_ordered"]:
            y_pred = torch.argmax(y_hat, dim=1).cpu().numpy()
            bacc = balanced_accuracy_score(y_np, y_pred)
            y_bin = label_binarize(y_np, classes=list(range(y_hat.shape[1])))
            roc_auc = roc_auc_score(y_bin, F.softmax(y_hat, dim=1).cpu().numpy(), multi_class="ovr")
            self.log(f"{prefix}/label_cat/bacc", bacc)
            self.log(f"{prefix}/label_cat/roc_auc", roc_auc)
            loss = F.cross_entropy(y_hat, y)
        else:
            raise ValueError(f"Unsupported target type: {target}")

        self.log(f"{prefix}/loss", loss)

        # --- Extended Metrics (WGA, Per-Label) ---
        extended_metrics = self._compute_detailed_metrics(y, y_hat, b, target)
        for k, v in extended_metrics.items():
            self.log(f"{prefix}/{k}", v)

        return loss

    def calculate_metrics(self, y, y_hat, b, target) -> dict:
        # Re-use logic for consistency
        metrics = {}
        y_np = y.cpu().numpy()
        y_hat_np = y_hat.cpu().numpy()

        if target == "label":
            y_pred = (y_hat_np > 0).astype(float)
            metrics["label/bacc"] = balanced_accuracy_score(y_np, y_pred)
            metrics["label/roc_auc"] = roc_auc_score(y_np, torch.sigmoid(y_hat).cpu().numpy())
        elif target == "label_c":
            metrics["label_c/mse"] = mean_squared_error(y_np, y_hat_np)
            metrics["label_c/r2"] = r2_score(y_np, y_hat_np)
        elif target in ["label_cat", "label_cat_ordered"]:
            y_pred = torch.argmax(y_hat, dim=1).cpu().numpy()
            metrics["label_cat/bacc"] = balanced_accuracy_score(y_np, y_pred)
            y_bin = label_binarize(y_np, classes=list(range(y_hat.shape[1])))
            metrics["label_cat/roc_auc"] = roc_auc_score(y_bin, F.softmax(y_hat, dim=1).cpu().numpy(), multi_class="ovr")

        # Add WGA and Per-Label
        extended = self._compute_detailed_metrics(y, y_hat, b, target)
        metrics.update(extended)
        
        return metrics

# --- Concrete Implementation ---

class MetaDataPrediction(MetaDataPredictionAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = self.get_loss_function(self.target, self.class_weights)
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.classifier_task(self.predictor(x)[1])

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        x = batch["img"]
        y = batch[self.target]
        
        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        l2_penalty = (y_hat.pow(2).sum(dim=1)).mean() if self.target == "label_cat" else 0.0
        self.manual_backward(loss + l2_penalty * 0.1)
        optimizer.step()

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["img"]
        y = batch[self.target]
        b = batch[self.protected_attr_key] # Grab bias using key (default "b")

        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        # Store Tuple: (Label, Prediction, Bias)
        self.validation_outputs.append((
            y.detach().cpu(), 
            y_hat.detach().cpu(), 
            b.detach().cpu()
        ))
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch["img"]
        y = batch[self.target]
        b = batch[self.protected_attr_key]

        y_hat = self.forward(x).squeeze()
        loss = self.loss(y_hat, y)

        self.test_outputs.append((
            y.detach().cpu(), 
            y_hat.detach().cpu(),
            b.detach().cpu()
        ))
        return loss

    def on_validation_epoch_end(self):
        # Unpack tuple of 3
        y = torch.cat([x[0] for x in self.validation_outputs])
        y_hat = torch.cat([x[1] for x in self.validation_outputs])
        b = torch.cat([x[2] for x in self.validation_outputs])

        self.calculate_log_metrics(y, y_hat, b, self.target, prefix="val")
        
        self.validation_outputs = []
        if not self.trainer.sanity_checking:
            if not self.lr_schedulers():
                return
            if isinstance(self.lr_schedulers(), list):
                for scheduler in self.lr_schedulers():
                    scheduler.step()
            else:
                self.lr_schedulers().step()

    def on_test_epoch_end(self):
        y = torch.cat([x[0] for x in self.test_outputs])
        y_hat = torch.cat([x[1] for x in self.test_outputs])
        b = torch.cat([x[2] for x in self.test_outputs])
        
        metrics = self.calculate_metrics(y, y_hat, b, self.target)
        self.test_outputs = []
        self.final_metrics = metrics
        return metrics




    


class cDiscoPredictor(MetaDataPrediction):
    def __init__(
        self,
        protected_attributes: list,  # e.g. ["cf"]
        bw: float = 1.0,
        cdcor_lambda: float = 1.0,
        warmup_ratio: float = 0.0,
        method: str = "max",
        *args,
        **kwargs,
    ):
        # Normalize method aliases to the canonical aggregation mode
        method_aliases = {
            "sdisco": "max",
            "sdisco_max": "max",
            "sdisco_mean": "mean",
            "standard": "max",
            "max": "max",
            "mean": "mean",
        }
        if method not in method_aliases:
            raise ValueError(
                f"Unsupported cdisco method: '{method}'. "
                f"Supported options: {list(method_aliases.keys())}"
            )
        agg_method = method_aliases[method]

        super().__init__(protected_attributes=protected_attributes, *args, **kwargs)
        self.bw = bw
        self.protected_attributes = protected_attributes
        self.max_cdcor_lambda = cdcor_lambda
        self.method = method
        self.warmup_ratio = warmup_ratio
        self.cdcor_func = partial(sdisco_metric, method=agg_method)

    @property
    def current_cdcor_lambda(self) -> float:
        """Dynamically computes the current lambda based on global step progress."""
        if self.warmup_ratio <= 0.0:
            return self.max_cdcor_lambda
        
        # This handles max_epochs, max_steps, gradient accumulation, and multi-GPU transparently
        total_steps = self.trainer.estimated_stepping_batches
        
        # In Lightning, if max_epochs = -1 and max_steps = -1, estimated_stepping_batches returns float('inf')
        if total_steps != float('inf') and total_steps > 0:
            # Calculate the total number of steps allocated to the warmup phase
            warmup_steps = max(1.0, total_steps * self.warmup_ratio)
            progress = self.global_step / warmup_steps
        else:
            # Fallback if training indefinitely without a defined end
            return self.max_cdcor_lambda

        # Clamp progress to 1.0 so it holds at the max value after the warmup phase
        return self.max_cdcor_lambda * min(1.0, float(progress))

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        x: torch.Tensor = batch["img"]  # new key
        y: torch.Tensor = batch[self.target]  # target is "label"
        y_hat: torch.Tensor = self.forward(x).squeeze()
        base_loss = self.loss(y_hat, y)
        z: torch.Tensor = batch[self.protected_attributes[0]]

        # l2 regularize y_hat outputs
        l2_penalty = (y_hat.pow(2).sum(dim=1)).mean() if self.target == "label_cat" else 0.0

        # y_hat softmax if target is label_cat
        y_hat = F.softmax(y_hat, dim=1) if self.target == "label_cat" else F.sigmoid(y_hat) if self.target == "label" else y_hat

        # # select ith element of y_hat where i is the index of y
        # b, p = y_hat.shape

        # # Use torch.arange to build batch indices
        # batch_indices = torch.arange(b)

        # # Use advanced indexing
        # y_hat = y_hat[batch_indices, y]

        # one hot encode y if it is label_cat
        if self.target == "label_cat":
            y = F.one_hot(y, num_classes=self.n_outputs).float()

        cdcor_loss = self.cdcor_func(y_hat, z, y, self.bw)

        loss = cdcor_loss * self.current_cdcor_lambda + base_loss + l2_penalty * 0.1

        self.manual_backward(loss)
        optimizer.step()

        self.log("train/loss", base_loss.detach().cpu())
        self.log("train/cdcor_loss", cdcor_loss.detach().cpu())
        self.log("train/full_loss", loss.detach().cpu())
        self.log("train/current_cdcor_lambda", self.current_cdcor_lambda)
        return loss

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["img"]
        y: torch.Tensor = batch[self.target]
        b = batch[self.protected_attr_key]
        z: torch.Tensor = batch[self.protected_attributes[0]]
        y_hat = self.forward(x).squeeze()
        base_loss = self.loss(y_hat, y)

        self.validation_outputs.append((y.detach().cpu(), y_hat.detach().cpu(), b.detach().cpu()))

        z: torch.Tensor = batch[self.protected_attributes[0]]  # e.g. "cf"

        # y_hat softmax if target is label_cat
        y_hat = F.softmax(y_hat, dim=1) if self.target == "label_cat" else F.sigmoid(y_hat) if self.target == "label" else y_hat
        
        # # select ith element of y_hat where i is the index of y
        # b, p = y_hat.shape

        # # Use torch.arange to build batch indices
        # batch_indices = torch.arange(b)

        # # Use advanced indexing
        # y_hat = y_hat[batch_indices, y]

        # one hot encode y if it is label_cat
        if self.target == "label_cat":
            y = F.one_hot(y, num_classes=self.n_outputs).float()

        cdcor_loss = self.cdcor_func(y_hat, z, y, self.bw)
        loss = cdcor_loss * self.current_cdcor_lambda + base_loss
        self.log("val/cdcor_loss", cdcor_loss.detach().cpu())
        self.log("val/loss", base_loss.detach().cpu())
        self.log("val/full_loss", loss.detach().cpu())
        return loss

