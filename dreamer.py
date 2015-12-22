import get_layer_data as gd
import dream_styles
import models as ml
import cv2
import setup_caffe_network as su


class Dreamer:
    def __init__(self):
        su.SetupCaffe.gpu_on()
        self.net = ml.NetModels.setup_googlenet_model('../CommonCaffe/TrainedModels/')

    iterator = [
        {
            'iter_n': 30,
            'start_sigma': 1.5,
            'end_sigma': 0.0,
            'start_step_size': 6.0,
            'end_step_size': 1.0
        },

    ]

    net = None
    stl = dream_styles.Styles()
    style_data = None
    subject_data = None
    layer = None

    __i_layer = 0

    def start_dreaming(self, frame):
        layers = ['inception_3b/5x5_reduce', 'inception_4a/5x5_reduce', 'inception_4c/3x3_reduce',
                  'inception_5a/3x3_reduce']
        self.layer = layers[self.__i_layer]
        self.__i_layer += 1
        if self.__i_layer >= len(layers):
            self.__i_layer = 0
        self.style_data = gd.get_layers_data(self.net, 'ImagesIn/elephants2.jpg', self.layer)
        self.subject_data = gd.get_layers_data_image(self.net, self.__input_filter(frame), self.layer)
        self.stl.setup_style_iterator(self.iterator[0])
        vis = self.stl.next_frame(self.net, self.style_data, self.subject_data, self.layer)
        return vis

    def get_dream_frame(self):
        return self.stl.next_frame(self.net, self.style_data, self.subject_data, self.layer)

    @staticmethod
    def __input_filter(img):
        img = cv2.medianBlur(img, 3)
        img = cv2.medianBlur(img, 3)
        img = cv2.medianBlur(img, 3)
        img = cv2.medianBlur(img, 5)
        img = cv2.bilateralFilter(img, 20, 50, 10)
        return img
