# OCT_EM_NN

This repository contains code related to creating and training a model to predict post-operative improvement in patients affected by Epiretinal Membrane.

## Model Architecture 

Both pre-operative OCT images and clinical data were available, therefore the model architecture is an hybrid between:

- A convolutional network to handle the OCT images
- Fully connected network to handle clinical data

The output layers from those subnets are concatenated and used as input for one last fully connected hidden layer before the target layer.

INSERIRE QUI IMMAGINE DELLA RETE

## Image preprocessing

The original image data from the OCT machine(?) is in form of a .tif file, containing both the tomography and an image of the retina with an arrow to highlight the angle from which the image was taken (then discarded), side by side.

![Alt text](OCT/02_0.tif)

Cropping and resizing was applied on every image original file to obtain 512x512 tomography images.
In the bottom left corner of most images there is a scale indicator. To verify which is the best approach, a similar architecture was also created to manage 444x444 images without an indicator.

## Data augmentation

## Multitarget loss

## Management of target imbalance

## Retraining with visus