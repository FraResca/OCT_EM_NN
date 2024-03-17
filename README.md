# OCT_EM_NN

This repository contains code related to creating and training a model to predict post-operative improvement in patients affected by Epiretinal Membrane.

## Model Architecture 

Both pre-operative OCT images and clinical data were available, therefore the model architecture is an hybrid between:
- A convolutional network to handle the OCT images
- Fully connected network to handle clinical data

The output layers from those subnets are concatenated and used as input for one last fully connected hidden layer before the target layer.

TODO: inserire immagine della rete

## Images preprocessing

The original image data from the OCT machine(?) is in form of a .tif file, containing both the tomography and an image of the retina with an arrow to highlight the angle from which the image was taken (then discarded), side by side.

![Alt text](ReadmeIMGS/OCT.png)

Cropping and resizing was applied on every image original file to obtain 512x512 tomography images.

![Alt text](ReadmeIMGS/OCTcrop.png)

In the bottom left corner of most images there is a scale indicator. A version of every image cropped to 444x444 was created for comparison reasons.

![Alt text](ReadmeIMGS/OCTnorul.png)

## Clinical data preprocessing

The pre-surgical clinical features from each patient chosen for training are:
- Sex (2 classes)
- Age (Numerical)
- Pucker stage (4 classes)
- Presence of cystoid edema (2 classes)
- Ellipsoid zone disruption (2 classes)
- Central subfield thickness (Numerical)
- Central thickness (Numerical)
- Phakic (2 classes)
- Diabetes (2 classes)

Along with patient-related features, surgical configuration information was added to the input data:
- Phaco/Cataract
- Tamponade

LabelEncoder from Sklearn was used to 

The post-surgical features chosen for learning are:
- Presence of cystoid edema (2 classes)
- Ellipsoid zone disruption (2 classes)
- Central subfield thickness (Numerical)
- Central thickness (Numerical)





## Data augmentation



## Multitarget loss

## Management of target imbalance

## Retraining with visus

TODO