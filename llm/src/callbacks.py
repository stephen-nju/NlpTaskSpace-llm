# coding:utf-8-----------

import copy
import os.path
from typing import TYPE_CHECKING, Dict

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy

if TYPE_CHECKING:
    import lightning as pl


class HFModelCheckpoint(ModelCheckpoint):
    # 重写model checkpoint 方案，直接保存hf模型
    def __init__(self, *args, **kwargs):
        self.save_hf = kwargs.pop("save_hf", False)
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        if self.save_hf:
            lightning_module = trainer.strategy.lightning_module
            if trainer.is_global_zero:
                lightning_module.model.save_pretrained(filepath)
                config_path = os.path.join(filepath, "config.json")
                if not os.path.exists(config_path):
                    lightning_module.model.config.save_pretrained(filepath)
        else:
            super()._save_checkpoint(trainer, filepath)

    # def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
    #     trainer.save_checkpoint(filepath, self.save_weights_only)

    #     self._last_global_step_saved = trainer.global_step
    #     self._last_checkpoint_saved = filepath

    #     # notify loggers
    #     if trainer.is_global_zero:
    #         for logger in trainer.loggers:
    #             logger.after_save_checkpoint(proxy(self))
