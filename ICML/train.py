from pathlib import Path
import wandb
import logging
import pandas as pd
from typing import List, Optional

import hydra
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from ICML.debiasing_methods import MetaDataPrediction, cDiscoPredictor

from utils.callbacks import EfficiencyTimingCallback


LOG = logging.getLogger(__name__)

def create_callbacks(config: DictConfig) -> List[pl.Callback]:
    callbacks = []
    if config.predictor.target in ["label", "label_cat", "label_cat_ordered"]:
        monitor = f"val/{config.predictor.target}/bacc"
        LOG.info("Creating ModelCheckpoint callback to monitor %s", monitor)
        callbacks.append(ModelCheckpoint(
            monitor=monitor,
            mode="max",
            filename="epoch-{epoch}-val_bacc-{" + f"{monitor}:.2f" + "}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            dirpath="chkpts",
            verbose=True,

        ))

        monitor2 = f"val/{config.predictor.target}/roc_auc"
        LOG.info("Creating ModelCheckpoint callback to monitor %s", monitor2)
        callbacks.append(ModelCheckpoint(
            monitor=monitor2,
            mode="max",
            filename="epoch-{epoch}-val_roc_auc-{" + f"{monitor2}:.2f" + "}",
            auto_insert_metric_name=False,
            save_top_k=1,
            dirpath="chkpts",
            verbose=True,
        ))

        if config.data.name != "yaleB":
            monitor3 = f"val/wga"
            LOG.info("Creating ModelCheckpoint callback to monitor %s", monitor3)
            callbacks.append(ModelCheckpoint(
                monitor=monitor3,
                mode="max",
                filename="epoch-{epoch}-val_wga-{" + f"{monitor3}:.2f" + "}",
                auto_insert_metric_name=False,
                save_top_k=1,
                dirpath="chkpts",
                verbose=True,
            ))

    elif config.predictor.target in ["label_c"]:
        monitor = f"val/{config.predictor.target}/mse"
        LOG.info("Creating ModelCheckpoint callback to monitor %s", monitor)
        callbacks.append(ModelCheckpoint(
            monitor=monitor,
            mode="min",
            filename="epoch-{epoch}-val_mse-{" + f"{monitor}:.2f" + "}",
            auto_insert_metric_name=False,
            save_top_k=1,
            dirpath="chkpts",
            save_last=True,
            verbose=True,
        ))

    else:
        raise ValueError(f"Unknown target {config.predictor.target} for ModelCheckpoint callback")
    return callbacks


def train(config: DictConfig):

    print(config.keys())

    try:
        predictor = config.get("predictor", None)
        debiasing_method = config.get("debiasing_method", None)
        config_callbacks = config.get("callbacks", {})
        config_loggers = config.get("logger", {})
        seed = config.get("seed", None)
        datamodule = config.data.get("datamodule", None)
        trainer = config.get("trainer", None)
    except AttributeError as e:
        raise ValueError("Config is not properly set up. Please check your config file.") from e

    # raise error if seed is not set
    if seed is None:
        raise ValueError("Seed must be set in the config file under 'seed'.")
    
    if predictor is None:
        raise ValueError("Predictor must be set in the config file under 'predictor'.")

    pl.seed_everything(seed, workers=True)

    LOG.info("Instantiating datamodule <%s>", datamodule._target_)
    data: pl.LightningDataModule = hydra.utils.instantiate(datamodule)

    if debiasing_method == "cdisco":
        LOG.info("Instantiating cDisco predictor <%s>", cDiscoPredictor)
        module: pl.LightningModule = cDiscoPredictor(**predictor)
    else:
        LOG.info("Instantiating baseline predictor <%s>", MetaDataPrediction)
        module: pl.LightningModule = MetaDataPrediction(**predictor)

    # Init lightning callbacks
    callbacks: List[pl.Callback] = []
    for cb_conf in config_callbacks.values():
        if "_target_" in cb_conf:
            LOG.info("Instantiating callback <%s>", cb_conf._target_)
            callbacks.append(hydra.utils.instantiate(cb_conf))

    callbacks += create_callbacks(config)

    # Add efficiency timing callback
    callbacks.append(EfficiencyTimingCallback())

    # Init lightning loggers
    logger: List[pl.LightningLoggerBase] = []
    for lg_conf in config_loggers.values():
        if "_target_" in lg_conf:
            LOG.info("Instantiating logger <%s>", lg_conf._target_)
            l = hydra.utils.instantiate(lg_conf)
            if isinstance(l, WandbLogger):
                # Log hyperparameters
                l.log_hyperparams(OmegaConf.to_container(config, resolve=True))
            logger.append(l)

    LOG.info("Instantiating trainer <%s>", trainer._target_)
    trainer: pl.Trainer = hydra.utils.instantiate(trainer, callbacks=callbacks, logger=logger)

    data.setup("fit")

    LOG.info("Starting training!")
    trainer.fit(module, data)

    wandb.finish()


@hydra.main(config_path="config", config_name="standard.yaml", version_base="1.3")
def main(config: DictConfig):
    output_dir = Path.cwd()  # Because hydra.job.chdir = True
    finished_flag = output_dir / "finished.txt"

    # Check if this experiment was already run
    if finished_flag.exists():
        print(f"[Hydra Skipper] Already completed: {output_dir}. Skipping...")
        return

    # ---- your training logic here ----
    print(f"[Hydra Runner] Running: {output_dir}")
    train(config)

    # Mark this run as completed
    finished_flag.write_text("done\n")

    return 1.0


if __name__ == "__main__":
    main()
