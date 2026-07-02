
Weapon Detection - v3 2022-09-16 4:16am
==============================

This dataset was exported via roboflow.com on September 16, 2022 at 1:24 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 4824 images.
Weapons are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 25 percent of the image
* Random rotation of between -19 and +19 degrees
* Random brigthness adjustment of between -25 and +25 percent
* Random exposure adjustment of between -18 and +18 percent
* Random Gaussian blur of between 0 and 2 pixels
* Salt and pepper noise was applied to 5 percent of pixels


