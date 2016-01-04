import numpy as np
import images
import cv2


class Styles:
    def __init__(self):
        pass

    @staticmethod
    def __blur(img, sigma):
        if sigma > 0.01:
            img[0] = cv2.GaussianBlur(img[0], (7, 7), sigma)
            img[1] = cv2.GaussianBlur(img[1], (7, 7), sigma)
            img[2] = cv2.GaussianBlur(img[2], (7, 7), sigma)
        return img

    @staticmethod
    def __gram_matrix(v):
        n_channels = v.shape[0]
        feats = np.reshape(v, (n_channels, -1))
        gram = np.dot(feats, feats.T)
        return gram

    # Modified from DeepDream ipython
    @staticmethod
    def __objective_guide(dst, subject, style):
        x = subject
        y = style
        ch = x.shape[0]
        x = np.reshape(x, (ch, -1))
        y = np.reshape(y, (ch, -1))
        A = np.dot(x.T, y)  # compute the matrix of dot-products with guide features

        dst.diff[0].reshape(ch, -1)[:] = y[:, A.argmax(1)]  # select ones that match best
        dst.diff[:] += dst.data  # this makes more network stuff appear

    def __gradient_ascent(self, src, sigma, step_size):
        # src = net.blobs['data']  # input image is stored in Net's 'data' blob
        g = src.diff[0]

        # apply normalized ascent step to the input image
        src.data[:] += step_size / np.abs(g).mean() * g

        src.data[0] = self.__blur(src.data[0], sigma)

    def __take_steps(self, net, sigma, step_size, style_data, subject_data, layer):
        top = layer

        net.forward(end=top)

        self.__objective_guide(net.blobs[top], subject_data, style_data)

        bottom = None
        net.backward(start=top, end=bottom)
        self.__gradient_ascent(net.blobs['data'], sigma, step_size)

    num_iterations = 1
    iteration = 0
    start_sigma = 0
    end_sigma = 0
    start_step_size = 2
    end_step_size = 1

    def __load_layer(self, a_layer):
        self.num_iterations = a_layer['iter_n']
        self.start_sigma = a_layer['start_sigma']
        self.end_sigma = a_layer['end_sigma']
        self.start_step_size = a_layer['start_step_size']
        self.end_step_size = a_layer['end_step_size']

    def __get_sigma(self, iteration):
        if iteration <= self.num_iterations:
            sig = self.start_sigma + ((self.end_sigma - self.start_sigma) * iteration) / self.num_iterations
        else:
            sig = self.end_sigma
        return sig

    def __get_step_size(self, iteration):
        if iteration <= self.num_iterations:
            step = self.start_step_size + ((self.end_step_size - self.start_step_size) * iteration) / self.num_iterations
        else:
            step = self.end_step_size
        return step

    def setup_style_iterator(self, iterator):
        self.iteration = 0
        self.__load_layer(iterator)

    def next_frame(self, net, style_data, subject_data, layer):
        im = images.Images()

        sigma = self.__get_sigma(self.iteration)
        step_size = self.__get_step_size(self.iteration)

        self.__take_steps(net, sigma, step_size, style_data, subject_data, layer)

        self.iteration += 1

        return im.visualize_src(net)


