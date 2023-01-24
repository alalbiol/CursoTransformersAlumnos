import warnings
from argparse import ArgumentParser
from pathlib import Path

import defusedxml

warnings.filterwarnings('ignore')
import pytorch_lightning as pl
# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from libs.callbacks import ImagePredictionLogger
from libs.data import Covid19DataModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim import SGD, Adam
from torchmetrics import AUROC, MeanMetric
from torchvision.models import ResNet50_Weights


# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ResNetClassifier(pl.LightningModule):
    def __init__(self,  resnet_version,
                optimizer='adam', lr=1e-3, 
                ):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            "resnet18": models.resnet18, "resnet34": models.resnet34,
            "resnet50": models.resnet50, "resnet101": models.resnet101,
            "resnet152": models.resnet152
        }
        # optimizers = {'adam': Adam, 'sgd': SGD}
        # self.optimizer = optimizers[optimizer]
        self.optimizer = optimizer
        self.lr = lr

        #instantiate loss criterion
        self.criterion = nn.BCEWithLogitsLoss() 

        self.rgb_adapter = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU())

        with torch.no_grad():
            self.rgb_adapter[0].weight.fill_(0) #type: ignore
            self.rgb_adapter[0].weight[:,:,1,1] = 1 #type: ignore

        

        # Using a pretrained ResNet backbone       
        if resnet_version == 50:
            self.resnet_model = resnets[resnet_version](weights = ResNet50_Weights)
        else:
            self.resnet_model = resnets[resnet_version](pretrained=True)

        # Replace old FC layer with Identity so we can train our own
        linear_size = self.resnet_model.fc.in_features

        # replace final layer for fine tuning
        num_classes = 1
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        # if tune_fc_only: # option to only tune the fully-connected layers
        #     for child in list(self.resnet_model.children())[:-1]:
        #         for param in child.parameters():
        #             param.requires_grad = False

        #metrics
        self.meanTrainLoss = MeanMetric()
        self.meanValLoss = MeanMetric()
        self.trainAUROC = AUROC(num_classes=1)
        self.valAUROC = AUROC(num_classes=1)


    def forward(self, X):
        X = self.rgb_adapter(X)
        return self.resnet_model(X)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return SGD(self.parameters(), lr=self.lr, weight_decay=1e-4, nesterov=True, momentum=0.9)
        else:
            raise ValueError("Optimizer not supported")
    
    
    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        preds = self(x)
        
        loss = self.criterion(preds.flatten(), y.float())

        with torch.no_grad():
            probs = torch.sigmoid(preds)
       
       
        #update metrics
        self.trainAUROC.update(probs.flatten(), y)
        self.meanTrainLoss.update(loss)
       
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

 
    
    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']

        preds = self(x)
        
        
        loss = self.criterion(preds.flatten(), y.float())
        
        with torch.no_grad():
            probs = torch.sigmoid(preds)
       
       
        #update metrics
        self.valAUROC.update(probs.flatten(), y)
        self.meanValLoss.update(loss)
    

        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss


    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        
        loss = self.criterion(preds, y)
        
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        

    def on_train_epoch_end(self) -> None:
        self.log("train_auroc", self.trainAUROC.compute(), on_epoch=True, prog_bar=True, logger=True)
        #self.log("train_mean_loss", self.meanTrainLoss.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.trainAUROC.reset()
        self.meanTrainLoss.reset()

        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.log("val_auroc", self.valAUROC.compute(), on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_mean_loss", self.meanValLoss.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.valAUROC.reset()
        self.meanValLoss.reset()

        return super().on_validation_epoch_end()


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument("-m", "--model",
                        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
                        type=str,default="resnet50",)
    
    parser.add_argument("--num_epochs", help="""Number of Epochs to Run.""", type=int, default=100)
    parser.add_argument("--train_set_csv", default = Path('text_files/train_set_articulo.csv'), help="""Path to training csv """, type=Path)
    parser.add_argument("--test_set_csv", default = Path('text_files/test_set_articulo.csv'),help="""Path to validation set folder.""", type=Path)

    parser.add_argument("--convert_RGB", default = False, help="""Convert images to RGB.""", type=bool)

    # Optional arguments
    parser.add_argument("-o", "--optimizer", help="""PyTorch optimizer to use. Defaults to adam.""", default='sgd', type=str)
    parser.add_argument("-lr", "--learning_rate", help="Adjust learning rate of optimizer.", type=float, default=1e-3)
    parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
                        type=int, default=16)

    parser.add_argument("-s", "--save_path", help="""Path to save model trained model checkpoint.""")
    parser.add_argument("-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=1)
    parser.add_argument("--wandb", default = True, type = bool, help="""Enables Weights and Biases logging.""")

    args = parser.parse_args()



    covidDataModule = Covid19DataModule( train_set_csv =  args.train_set_csv,
                                        test_set_csv = args.test_set_csv,
                                        batch_size = args.batch_size,
                                        convert_RGB = args.convert_RGB,
                                        )       
                
    # # Instantiate Model
    model = ResNetClassifier( resnet_version = args.model,
                            optimizer = args.optimizer, lr = args.learning_rate,
                            )
    # Instantiate lightning trainer and train model
    trainer_args = {'gpus': args.gpus, 
                'max_epochs': args.num_epochs}



    # callbacks section
    callbacks = []

    from pytorch_lightning.callbacks import ModelCheckpoint

    #callbacks.append(ModelCheckpoint(monitor='val_auroc', mode='max', save_top_k=1, save_last=True, filename='{epoch}-{val_auroc:.2f}'))
    #trainer_args['callbacks'] = 

    if args.wandb:
        trainer_args['logger'] = WandbLogger(name='ResNet50_sgd_data_articulo', project='Covid19')
        callbacks.append(ImagePredictionLogger(num_samples=16))
    

    trainer = pl.Trainer(  **trainer_args, callbacks = callbacks)
    trainer.fit(model, datamodule=covidDataModule)
    # Save trained model
    save_path = (args.save_path if args.save_path is not None else './') + 'trained_model.ckpt'
    trainer.save_checkpoint(save_path)