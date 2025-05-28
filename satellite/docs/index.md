# Segmentation model

## Training dataset analysis

These are the available channels for one image:
![dataset_channels](images/dataset_channels.png)

We want to identify the clouds on RGB channels thanks to the GT mask.

This is a compiled RGB image (this will be used as input for the model):

![dataset_rgb_composite](images/dataset_rgb_composite.png)

This is a compiled RGBA image with clouds removed (this will be the postprocessed output image):

![dataset_rgba_composite](images/dataset_rgba_composite_alpha_inverted.png)

## Train custom U-net

Results for a first training of 10 epochs:

![training_10_epochs](images/training_10_epochs_0.png)
![training_10_epochs](images/training_10_epochs_1.png)
![training_10_epochs](images/training_10_epochs_2.png)

## Download and analyse Sentinel-2 data

On télécharge les données depuis le bucket S3 public.
Ce sont des images en JPEG2000, nous devons donc utiliser `rasterio` plutôt que `Pillow`.

`rasterio` est fait pour:

- Lecture GeoTIFF / JP2
- Métadonnées géospatiales (CRS, bbox), accède aux coordonnées, UTM, - projection, pas `Pillow`
- Images multi-bandes (ex : 13 bandes) gère parfaitement les stacks de bandes, `Pillow` ne comprend que RGB / L / P
- Support NIR, SWIR, bandes 16-bit+ alors que `Pillow` est limité à 8-bit
- Interopérabilité SIG (QGIS, GDAL)

|channel|sentinel code|
|-|-|
|R|B04|
|G|B03|
|B|B02|
|NIR|B08|

![sentinel_raw_data_0](images/sentinel_raw_data_0.png)
![sentinel_raw_data_0](images/sentinel_raw_data_1.png)

## Mosaic

![mosaic](images/mosaic.png)
