import warnings
from argparse import ArgumentParser
from pathlib import Path

import defusedxml

warnings.filterwarnings('ignore')
from transformers import ViTForImageClassification, AdamW
import torch.nn as nn

from pytorch_lightning.loggers.wandb import WandbLogger 



import pytorch_lightning as pl
# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from libs.data import Covid19DataModule
from torch.optim import SGD, Adam
from torchmetrics import AUROC
from torchmetrics import  MeanMetric

from libs.callbacks import ImagePredictionLogger


id2label = {0: 'non-COVID-19', 1: 'COVID-19'}
label2id = {'COVID-19': 1, 'non-COVID-19': 0}





# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ViTClassifier(pl.LightningModule):
    def __init__(self,
                optimizer='adam', lr=1e-3, 
                ):
        super().__init__()

        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=2,
                                                              id2label=id2label,
                                                              label2id=label2id)
        # optimizers = {'adam': Adam, 'sgd': SGD}
        # self.optimizer = optimizers[optimizer]
        self.optimizer = optimizer
        self.lr = lr

        #instantiate loss criterion
        self.criterion  = nn.CrossEntropyLoss()



        #metrics
        self.trainAUROC = AUROC(num_classes=1)
        self.valAUROC = AUROC(num_classes=1)


    def forward(self, X):
        #expects three channel image
        outputs = self.vit(pixel_values=X) #type: ignore
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['image']
        labels = batch['label']
        logits = self(pixel_values)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
        
        

        
        loss = self.criterion(logits, labels.long())
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy, probs[:,1], labels

    def training_step(self, batch, batch_idx):
        loss, accuracy, probs, labels = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)
        self.trainAUROC.update(probs.flatten(), labels.flatten())

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy, probs, labels = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        self.valAUROC.update(probs.flatten(), labels.flatten())

        return loss



    def configure_optimizers(self):
        if self.optimizer == "adam":
            return Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return SGD(self.parameters(), lr=self.lr, weight_decay=1e-4, nesterov=True, momentum=0.9)
        elif self.optimizer == "adamw":
            return AdamW(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Optimizer not supported")
    
    
        

    def on_train_epoch_end(self) -> None:
        self.log("train_auroc", self.trainAUROC.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.trainAUROC.reset()
    
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.log("val_auroc", self.valAUROC.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.valAUROC.reset()
    
        return super().on_validation_epoch_end()


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    
    parser.add_argument("--num_epochs", help="""Number of Epochs to Run.""", type=int, default=100)
    parser.add_argument("--train_set_csv", default = Path('text_files/train_set_articulo.csv'), help="""Path to training csv """, type=Path)
    parser.add_argument("--test_set_csv", default = Path('text_files/test_set_articulo.csv'),help="""Path to validation set folder.""", type=Path)

    parser.add_argument("--convert_RGB", default = True, help="""Convert images to RGB.""", type=bool)

    # Optional arguments
    parser.add_argument("-o", "--optimizer", help="""PyTorch optimizer to use. Defaults to adam.""", default='adamw', type=str)
    parser.add_argument("-lr", "--learning_rate", help="Adjust learning rate of optimizer.", type=float, default=5e-5)
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
                                        imsize= 224,
                                        )       
                
    # # Instantiate Model
    model = ViTClassifier( optimizer = args.optimizer, lr = args.learning_rate,
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
        trainer_args['logger'] = WandbLogger(name=f'ViT_{args.optimizer}_data_articulo', project='Covid19')
        #callbacks.append(ImagePredictionLogger(num_samples=16))
    

    trainer = pl.Trainer(  **trainer_args, callbacks = callbacks)
    trainer.fit(model, datamodule=covidDataModule)
    # Save trained model
    # save_path = (args.save_path if args.save_path is not None else './') + 'trained_model.ckpt'
    # trainer.save_checkpoint(save_path)