import numpy as np

class MaxPoolLayer:

    def __init__(self, stride, pool_size, input_img):
        self.stride = stride
        self.pool_size = pool_size
        self.input_img = input_img
        self.im_height, self.im_width, self.im_depth = input_img.shape
        self.out_height = int((self.im_height-pool_size)/stride)+1
        self.out_width = int((self.im_width-pool_size) /stride)+1
        self.output = np.zeros((self.out_height, self.out_width, self.im_depth))

    def maxpool(self):
        for i in range(self.im_depth):
            curr_y = out_y = 0
            while curr_y + self.pool_size <= self.im_height:
                curr_x = out_x = 0
                while curr_x + self.pool_size <= self.im_width:
                    self.output[out_y, out_x, i] = np.max(self.input_img[curr_y:curr_y + self.pool_size, curr_x:curr_x + self.pool_size, i])
                    curr_x += self.stride
                    out_x += 1
                curr_y += self.stride
                out_y += 1