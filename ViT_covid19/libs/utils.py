import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont, Image


def show_examples_dataset(ds, examples_per_class: int = 2, size=(512, 512)):
    # dataset must implement properties labels_names, get_all_labels


    w, h = size
    labels = ds.label_names
    # grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    # draw = ImageDraw.Draw(grid)
    # font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 24)

    
    #make subplots of examples_per_class x len(labels)
    fig, axs = plt.subplots(len(labels), examples_per_class)

    all_labels_ds = ds.get_all_labels

    for label_id, label in enumerate(labels):

        # Filter the dataset by a single label, shuffle it, and grab a few samples
        sample_idx = all_labels_ds.loc[all_labels_ds == label].sample(examples_per_class)

        

        # Plot this label's examples along a row
        for i, example_idx in enumerate(sample_idx.index):
            sample = ds[example_idx]



            image =sample['image'][0].numpy() #gray image

            print("image min max",image.min(),image.max())

            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class , idx // examples_per_class )
            axs[box[1], box[0]].imshow(image, cmap='gray')
            axs[box[1], box[0]].set_title(label)
            axs[box[1], box[0]].axis('off')
            axs[box[1], box[0]].set_aspect('equal')
            #set the tight layout
            fig.tight_layout()            


            # grid.paste(image.resize(size), box=box)
            # draw.text(box, label, (255, 255, 255), font=font)

    return fig