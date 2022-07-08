from conv2d import ConvolutionLayer
from maxpool import MaxPoolLayer
from fullyconnected import FullyConnectedLayer
import numpy as np
import matplotlib.pyplot as plt

def save_image(image_num,layer):
    for i in range(layer.filt_num):
        plt.figure()
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.imshow(layer.output[:,:,i].reshape(layer.out_size,layer.out_size), cmap=plt.cm.binary)
        plt.xlabel(f'Image {image_num} after Convolution Layer {layer.layer_num} Filter {i}')
        plt.savefig(f'post_conv_img/img{image_num}_conv{layer.layer_num}_filt{i}')
        plt.close()

def predict(image,image_num,f1, f2, w3, w4, b1, b2, b3, b4, conv_s = 1, pool_f = 2, pool_s = 2):

    #Sample network from TensorFlow Trained MNIST CNN weights
    Conv1_pad = ConvolutionLayer(weights=f1, stride=conv_s, bias=b1, input_img=image, padding=True, layer_num=0)
    Conv1_pad.conv()
    save_image(image_num,Conv1_pad)
    Conv1_pad.relu()
    save_image(image_num,Conv1_pad)

    Conv2_pad = ConvolutionLayer(weights=f2, stride=conv_s, bias=b2, input_img=Conv1_pad.output, padding=True, layer_num=1)
    Conv2_pad.conv()
    save_image(image_num,Conv2_pad)
    Conv2_pad.relu()
    save_image(image_num,Conv2_pad)

    Maxpool = MaxPoolLayer(stride=pool_s, pool_size=pool_f, input_img=Conv2_pad.output)
    Maxpool.maxpool()

    Dense1 = FullyConnectedLayer(w3, b3, Maxpool.output)
    Dense1.dense()
    Dense1.relu()

    Dense2 = FullyConnectedLayer(w4, b4, Dense1.output)
    Dense2.dense()
    Dense2.softmax()

    return np.argmax(Dense2.output), np.max(Dense2.output)