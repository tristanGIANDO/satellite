# Training

## Create a custom U-net

A U-Net model is a type of convolutional neural network (CNN) designed specifically for image segmentation tasks.

There are two main parts:

1. Encoder:
   Convolution series + max pooling which progressively reduces the image resolution while increasing the number of channels to understand the image context.

2. Decoder:
   Series of transposed convolutions (or upsampling) that increase resolution.

The result is a binary mask. In our case, this will be the pixels that correspond to clouds.

### Results

Results for a first training of 10 epochs with only 100 images:

![training_10_epochs](images/training_10_epochs_0.png)
![training_10_epochs](images/training_10_epochs_1.png)
![training_10_epochs](images/training_10_epochs_2.png)
