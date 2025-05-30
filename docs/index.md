# Home

## Pipeline

1. Download Sentinel-2 images from the same location on several dates.
2. Run the cloud segmentation model and generate a “cloud mask”.
3. From this mask, remove images that are too cloudy from the dataset.
4. Transform the mask into alpha (invert it to remove the remaining clouds).
5. Stack the RGBA images to build a cloud-free mosaic.