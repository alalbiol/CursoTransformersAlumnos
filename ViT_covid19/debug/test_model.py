import sys

import matplotlib.pyplot as plt
sys.path.append('../')

from libs.data import get_covid_dataloaders
from libs.pl_model import Covid19Model


model = Covid19Model()




batch = 8
train_set_csv = '../text_files/train_set_articulo.csv'
test_set_csv = '../text_files/test_set_articulo.csv'
trainDL, testDL =  get_covid_dataloaders( batch, 
            train_set_csv,
            test_set_csv,
            root_images = '/home/alalbiol/Data/all_fisabio_256_photometric_fixed', 
            photometric_fixed = True,
            imsize = 256,)


for batch in trainDL:
    print(batch['image'].shape)
    print(batch['label'].shape)

    logits = model(batch['image'])
    print(logits.shape)
    break