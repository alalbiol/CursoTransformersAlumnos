# lightning related imports
import pytorch_lightning as pl
import torch
import wandb

from pytorch_lightning.callbacks import Callback

from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Optional

#from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY


from pytorch_lightning.utilities import rank_zero_info




#@CALLBACK_REGISTRY
class ImagePredictionLogger(Callback):
    def __init__(self, num_samples: int =32, epoch_interval: int  = 1):
        super().__init__()
        self.num_samples = num_samples
        self.epoch_inverval = epoch_interval
        
   
    def log_batch_images(self, batch, pl_module, trainer, preffix = "train"):

        imgs = batch['image'].to(device=pl_module.device)
        labels = batch['label'].to(device=pl_module.device)
       
        y_logits = pl_module(imgs)
        y_scores = torch.sigmoid(y_logits)
        preds = (y_scores > 0.5).int()

        imgs = imgs.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        
        trainer.logger.experiment.log({
            f"{preffix}_samples":[wandb.Image(x[0], caption=f"Pred:{pred.item()}, Label:{y}") 
                        for x, pred, y in zip(imgs[:self.num_samples], 
                                                preds[:self.num_samples], 
                                                labels[:self.num_samples])]
            })

    def on_validation_batch_end(self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs: Any,
    ) -> None:
        """Called when the validation batch ends."""
        
        if batch_idx == 0 and pl_module.current_epoch % self.epoch_inverval == 0:
            self.log_batch_images(batch, pl_module, trainer, preffix = "val")
            

        
    def on_train_batch_end(self, 
                        trainer: "pl.Trainer", 
                        pl_module: "pl.LightningModule", 
                        outputs: STEP_OUTPUT, 
                        batch: Any, 
                        batch_idx: int,
                        *args,
                        **kwargs) -> None:
        """Called when the train batch ends."""
        
        if batch_idx == 0 and pl_module.current_epoch % self.epoch_inverval == 0:
            self.log_batch_images(batch, pl_module, trainer, preffix = "train")
    
