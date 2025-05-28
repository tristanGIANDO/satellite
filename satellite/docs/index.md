# The Dataset

These are the available channels for one image:
![dataset_channels](images/dataset_channels.png)

We want to identify the clouds on RGB channels thanks to the GT mask.

This is a compiled RGB image (this will be used as input for the model):
![dataset_rgb_composite](images/dataset_rgb_composite.png)

This is a compiled RGBA image with clouds removed (this will be the postprocessed output image)
![dataset_rgba_composite](images/dataset_rgba_composite_alpha_inverted.png)