import multiprocessing
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import transforms

# import torch dataloader


RX_to_tensor = transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32) / 255.0 ).unsqueeze(0))
Norm_max = transforms.Lambda(lambda image: image / (image.max()+ 1e-6))
ToRGB = transforms.Lambda(lambda image: image.repeat(3,1,1))

def ConvertPIL8u(X):
    X = np.array(X).astype(np.float32)
    X=(X * 255 / np.max(X)).astype('uint8')
    out = Image.fromarray(X).convert('L')
    return out



DATAROOT = '/home/alalbiol/Data/all_fisabio_256_photometric_fixed'


from libs.augmentations import RandAugment


def load_image(row,root_images, photometric_fixed):
    # pos_path = '/media/HD/fisabio'
    # neg_path = '/media/HD/fisabio_covid_neg/covid19_neg'
    #pos_path = '/home/alalbiol/Data/all_fisabio_256'
    #neg_path = '/home/alalbiol/Data/all_fisabio_256'
    #photometric_fixed = False

    #pos_path = '/home/alalbiol/Data/all_fisabio_256_photometric_fixed'
    #neg_path = '/home/alalbiol/Data/all_fisabio_256_photometric_fixed'
    #photometric_fixed = True
    pos_path = root_images
    neg_path = pos_path
    #logger.debug(f'pos path: {pos_path} neg path {neg_path}')
    #print(f'pos path: {pos_path} neg path {neg_path}')

    
    file_path = pos_path if row['label']=='covid_pos' else neg_path
    im =  Image.open(os.path.join(file_path, row['full_path']))
    im = np.array(im)
    max_im = im.max()

    if photometric_fixed == False:
        if row['photometric_interp'] == "MONOCHROME1":
            # print("converting ", im.max())
            if max_im < 5000:
                #print("12 bit")
                im = 4095-im
            elif max_im < 35000:
                #print("16 bit")
                im = 32767-im
            else:
                im = 65535-im
    return Image.fromarray(im)




class CovidDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, load_image_fn = load_image, transform=None, training = False,root_images = '/home/alalbiol/Data/all_fisabio_256_photometric_fixed', 
            photometric_fixed = True, imsize = 256):
        """
        Args:
            csv_file (string): Path to the csv file
            load_image_fn (function): Function to read one image of the csv file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = pd.read_csv(csv_file)
        # if len(self.images)> 1000:
        #     self.images = self.images.sample(1000)
        self.load_image_fn = load_image_fn
        self.transform = transform
        self.targets = list(self.images.label.apply(lambda label: 0 if label=='covid_neg' else 1))
        self.training = training
        self.root_images = root_images
        self.photometric_fixed = photometric_fixed

 
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
       
        #row = self.images.loc[self.images.norm_filename == idx].iloc[0]
        row = self.images.iloc[idx]


        image = self.load_image_fn(row,self.root_images, self.photometric_fixed)

        label = 0 if row['label']=='covid_neg' else 1  # type: ignore
        id = row['norm_filename']  # type: ignore
        label_name = row['label'] # type: ignore

        if self.training:
            label_name = 'covid_neg' if label == 0 else 'covid_pos'



        if self.transform:
            image = self.transform(image)


        sample = {'image': image, 'label': label,'id': id,'label_name':label_name}



        return sample


    def display(self,idx):
        sample = self[idx]
        plt.imshow(sample['image'][0],cmap='gray')
        plt.title('covid_pos' if sample['label']==1 else 'covid_neg')


    @property
    def label_names(self):
        return ["covid_neg", "covid_pos"]

    @property
    def get_all_labels(self):
        return self.images['label']


def get_covid_dataloaders( batch, 
            train_set_csv,
            test_set_csv,
            root_images = DATAROOT, 
            photometric_fixed = True,
            imsize = 256,):
    
    transform_train = transforms.Compose([
        transforms.Resize(imsize),
        ConvertPIL8u,
        RandAugment(4,30),
        RX_to_tensor,
        #Norm_max
    ])

    transform_test = transforms.Compose([
        transforms.Resize(imsize),
        ConvertPIL8u,
        RX_to_tensor,
        #Norm_max
    ])


    
    print(f"Trainset csv {train_set_csv}")
    print(f"Testset csv {test_set_csv}")

    trainset = CovidDataset(train_set_csv,transform = transform_train,
                    photometric_fixed = photometric_fixed, root_images = root_images)
    #total_trainset = CovidDataset(os.path.join(csv_files_path, 'train_set_articulo_extended.cvs'),transform = transform_train,training=True)
    testset =  CovidDataset(test_set_csv,transform = transform_test,
                     photometric_fixed = photometric_fixed, root_images = root_images)


 
    print(f"len total trainset =   {len(trainset)}")
    print(f"len total testset =   {len(testset)}")

 
    trainloader = DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=8, pin_memory=True,
        drop_last=True)
    
    testloader = DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True,
        drop_last=False
    )
    return  trainloader, testloader




class Covid19DataModule(pl.LightningDataModule):
    def __init__(self, train_set_csv = 'text_files/train_set_articulo.csv' , 
                test_set_csv = 'text_files/test_set_articulo.csv',
                batch_size: int =32,  root_images: str= DATAROOT, convert_RGB = False,
                num_workers = -1, imsize = 256, **kwargs):
        super().__init__()

        print("Options in DataModule", kwargs)

        self.batch_size = batch_size
        self.root_images = root_images
        self.num_workers = num_workers if num_workers > 0 else multiprocessing.cpu_count()-1
        

        photometric_fixed = kwargs.get('photometric_fixed',True)

        transform_train = [
                transforms.Resize(imsize),
                ConvertPIL8u,
                RandAugment(6,30),
                RX_to_tensor,
            ]

        transform_test = [
                transforms.Resize(imsize),
                ConvertPIL8u,
                RX_to_tensor,
            ]

        if convert_RGB:
            transform_train.append(ToRGB)
            transform_test.append(ToRGB)
        
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        
        print(f"Trainset csv {train_set_csv}")
        print(f"Testset csv {test_set_csv}")

        self.train_dataset  = CovidDataset(train_set_csv,transform = transform_train,
                        photometric_fixed = photometric_fixed, root_images = root_images)
        #total_trainset = CovidDataset(os.path.join(csv_files_path, 'train_set_articulo_extended.cvs'),transform = transform_train,training=True)
        self.val_dataset  =  CovidDataset(test_set_csv,transform = transform_test,
                        photometric_fixed = photometric_fixed, root_images = root_images)


    
        print(f"len total trainset =   {len(self.train_dataset )}")
        print(f"len total testset =   {len(self.val_dataset )}")


            
        self.num_classes = 2

        self.save_hyperparameters()

    def prepare_data(self):
        if not pathlib.Path(self.root_images).exists():
            print("Get fisabio covid19 dataset")

    def setup(self, stage=None):
        # build dataset
        # caltect_dataset = ImageFolder('Caltech101')
        # # split dataset
        # self.train, self.val, self.test = random_split(caltect_dataset, [6500, 1000, 1645])
        # self.train.dataset.transform = self.augmentation
        # self.val.dataset.transform = self.transform
        # self.test.dataset.transform = self.transform
        #print("Nothing to do in setup datasets, partitions already given")
        return None
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,  num_workers=self.num_workers, drop_last=False,shuffle=False)

    @staticmethod
    def add_model_specific_args(parser):
        #parser = parent_parser.add_argument_group("model")
        #parser.add_argument("--data.train_set_csv", type=str, default='text_files/train_set_articulo.csv')
        #parser.add_argument("--data.test_set_csv", type=str, default='text_files/test_set_articulo.csv')
        return parser

    