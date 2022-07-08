import numpy as np

class ConvolutionLayer:

    def __init__(self, weights, stride, bias, input_img, padding, layer_num):
        self.weights = weights
        self.stride = stride
        self.bias = bias
        self.input_img = input_img
        self.padding = padding
        self.layer_num = layer_num
        self.pad_num = 0
        (self.filt_size, _, self.filt_depth, self.filt_num) = self.weights.shape
        self.im_size, _, self.im_depth = self.input_img.shape

        if padding and self.stride==1:
            self.out_size = self.im_size
            self.pad_num = int((self.stride*(self.out_size-1)-self.im_size+self.filt_size)/2)
            pad_image = np.zeros((self.im_size+self.pad_num*2, self.im_size+self.pad_num*2, self.im_depth))
            org_img_size,_,org_img_depth = self.input_img.shape

            #creating padded image
            for curr_depth in range(self.im_depth):
                for im_i in range(org_img_size):
                    for im_j in range(org_img_size):
                        pad_image[im_i+self.pad_num][im_j+self.pad_num][curr_depth] = self.input_img[im_i][im_j][curr_depth]

            self.input_img = pad_image
            self.im_size, _, self.im_depth = self.input_img.shape
            del pad_image, org_img_size, org_img_depth

        else:
            self.out_size = int((self.im_size - self.filt_size)/self.stride)+1

        self.output = np.zeros((self.out_size, self.out_size, self.filt_num))

    def details(self):
        return f"Filter size: {self.filt_size} x {self.filt_size}, Filter depth: {self.filt_depth}, Number of Filters: {self.filt_num}"

    def conv(self):
        assert self.im_depth == self.filt_depth, "Checking whether image depth equals to filter depth"

        for current_filter in range(self.filt_num):
            curr_row = out_row = 0
            while curr_row + self.filt_size <= self.im_size:
                curr_col = out_col = 0
                while curr_col + self.filt_size <= self.im_size:
                    self.output[out_row, out_col, current_filter] = np.sum(self.weights[:,:,:,current_filter] * self.input_img[curr_row:curr_row+self.filt_size, curr_col:curr_col+self.filt_size, :]) + self.bias[current_filter]
                    curr_col += self.stride
                    out_col += 1
                curr_row += self.stride
                out_row += 1
    
    def relu(self):
        self.output = self.output * (self.output > 0)