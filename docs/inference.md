# Inference and post-processing

The goal is to:

1. Give an image to the model so that it can identify cloudy areas.
2. Remove images that are too cloudy, based on the percentage of white pixels in the predicted mask.
3. Invert the mask and transform it into alpha, then build an RGBA image.
4. Create a mosaic by assembling several images of the same area but at different dates.

## Create the mosaic

![mosaic](images/mosaic.png)
