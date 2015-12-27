import dream_styles
import setup_caffe_network as su
import models as ml
import get_layer_data as gd
import cv2
import display
import dreamer
import images

print(cv2.__version__)

iterator = [
    {
        'iter_n': 40,
        'start_sigma': 1.5,
        'end_sigma': 0.0,
        'start_step_size': 6.0,
        'end_step_size': 1.0
    },

]


layer = 'inception_5a/3x3_reduce'

# su.SetupCaffe.gpu_on()
net = ml.NetModels.setup_googlenet_model('../CommonCaffe/TrainedModels/')

style_data = gd.get_layers_data(net, 'ImagesIn/724 pirate ship.jpg', layer)
img = cv2.imread('Paintings/BigSue53.jpg')
img = images.Images.resize_image(480, 640, img)
dreamer.Dreamer.input_filter(img)
subject_data = gd.get_layers_data_image(net, img, layer)

stl = dream_styles.Styles()

stl.setup_style_iterator(iterator[0])

for i in range(0, iterator[0]['iter_n']):
    vis = stl.next_frame(net, style_data, subject_data, layer)
    cv2.imshow('Video', vis)
    cv2.waitKey(1)

vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
display.Display().showResultPIL(vis)

