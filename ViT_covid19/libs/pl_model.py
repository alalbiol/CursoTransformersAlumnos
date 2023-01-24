from distutils.command.config import config

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.utilities import rank_zero_info
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchmetrics import (AUROC, Accuracy,  MeanMetric)
from torchvision import models

# from utils.looses import get_loss_fn
# from utils.metrics import MultipleEpochMetrics

from libs.backbone_models import ResNetTorch

#  --- Pytorch-lightning module ---


class Covid19Model(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet50",
        train_bn: bool = True,
        milestones: tuple = (50, 100),
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        loss_name = "binary_ce",
        num_workers: int = 6,
        num_classes: int = 1,
        metrics: list = [],
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor
            train_bn: Whether the BatchNorm layers should be trainable
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.loss_name = loss_name

        self.__build_model()

        self.meanTrainLoss = MeanMetric()
        self.meanValLoss = MeanMetric()
        self.trainAUROC = AUROC(num_classes=1)
        self.valAUROC = AUROC(num_classes=1)


        
        self.save_hyperparameters()

        

    def __build_model(self):
        """Define model layers & loss."""

        if "resnet50" == self.backbone:
            self.model = ResNetTorch(num_classes = 1)
        else:
            raise NameError(f"Backbone {self.backbone = } not implemented")

        # 3. Loss:
        
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        """Forward pass.
        Returns logits.
        """
        # 1. Feature extraction:
        x = self.model(x)   
        #(returns logits):
        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits.flatten(), target=labels.flatten().float())

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        image = batch['image']
        y_true = batch['label']


        y_logits = self.forward(image)
        y_probs = torch.sigmoid(y_logits)

        # 2. Compute loss
        loss = self.loss(y_logits, y_true)

        if loss.isnan():
            print("y_logits", y_logits)
            print("y_true", y_true)
            raise ValueError("Loss is NaN!")
    
       
         # 3. Compute metrics and log loss:
        self.log("train_loss", loss, prog_bar=False)

        self.meanTrainLoss.update(loss)
        self.trainAUROC.update(y_probs.flatten(), y_true.flatten())



        return loss


    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        image = batch['image']
        y_true = batch['label'].float()



        y_logits = self.forward(image)
        y_probs = torch.sigmoid(y_logits)
        
    

        # 2. Compute loss 
        loss = self.loss(y_logits, y_true)


         # 3. Compute metrics and log loss:
        self.log("val_loss", loss, prog_bar=True)
        self.meanValLoss.update(loss)
        self.valAUROC.update(y_probs.flatten(), y_true.long().flatten())
        

        return loss

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)

        #optimizer = optim.SGD(trainable_parameters, lr=self.lr, momentum=0.9, weight_decay=1e-4)
        #scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.5, patience=5)

        print("Using MultiStepLR with milestones", self.milestones)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor":"val_epoch_loss"}

    def training_epoch_end(self, training_step_outputs):

        # compute and log epoch metrics
        # train_accuracy = self.train_acc.compute()
        # get the mean loss over all batches

        

        # log the mean loss
        train_epoch_loss = self.meanTrainLoss.compute()
        self.log("train_epoch_loss", train_epoch_loss)
        self.log("train_epoch_auroc", self.trainAUROC.compute())
        
        
        # reset all metrics
        self.meanTrainLoss.reset()
        self.trainAUROC.reset()
        


    def validation_epoch_end(self, validation_step_outputs):
        # compute and log metrics 

        
        val_epoch_loss = self.meanValLoss.compute()

        self.log("val_epoch_loss", val_epoch_loss)
        self.log("val_epoch_auroc", self.valAUROC.compute())
    
        # reset all metrics
        self.meanValLoss.reset()
        self.valAUROC.reset()
            
        sch = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.

        # if self.current_epoch: #not first epoch
        #     sch.step()

        # if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #     sch.step(results_patient['cohenkappa_patient'])
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        #parser.add_argument("--model.encoder_layers", type=int, default=12)
        #parser.add_argument("--model.data_path", type=str, default="/some/path")
        return parent_parser

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    rank_zero_info(f"Gradient for {name} is not valid")
                    break

        if not valid_gradients:
            rank_zero_info(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()