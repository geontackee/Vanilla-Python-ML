# Vanilla-Python-ML
Importing Weights from Tensorflow Model and running it on Vanilla Python. Modified Alejandro Escontrela's Numpy-CNN repo to support weights from TensorFlow. Modified and Written by Geon Tack Lee. No training supported. Supports exporting of images after each convolution layer

# How to run:
1. Run export.py and retrieve a pkl file with the trained TensorFlow model weights.
2. Adjust networks.py file's predict() function with corresponding TensorFlow model network structure. If you want to export images, run save_image() after convolution on the ConvolutionLayer object.
3. Measure performance of trained TensorFlow model by running `python3 measure_performance.py '<weights_file_name>.pkl' '<test_images_file_name>.gzip' '<test_labels_file_name>.gzip'`
