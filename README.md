# OCT_EM_NN

This repository contains code related to creating and training a model to predict post-operative improvement in patients affected by Epiretinal Membrane.

## Model Architecture 

Both pre-operative OCT images and clinical data were available, therefore the model architecture is hybrid:

- Convolutional network to handle the OCT images
- Fully connected network to handle clinical data

The output layers from those modules are concatenated and used as input for one last fully connected hidden layer before the target layer.

INSERIRE QUI IMMAGINE DELLA RETE

## Image preprocessing

The original image data from the OCT machine(?) is in form of a .tif file, containing both the tomography and an image of the retina with an indicator of the angle from which the image was taken (the discarded), side by side. Cropping and resizing was applied on every image to obtain 512x512 tomography images.
In the bottom left corner of most images there is a scale indicator. To verify which is the best approach, a similar architecture was also created to manage 444x444 images without an indicator.

## Data augmentation

## Multitarget loss

## Management of target imbalance

## Retraining with visus