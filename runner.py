import dream_styles
import setup_caffe_network as su
import models as ml
import get_layer_data as gd
import cv2
import display

iterator = [
    {
        'iter_n': 30,
        'start_sigma': 6.0,
        'end_sigma': 0.0,
        'start_step_size': 6.0,
        'end_step_size': 1.0
    },

]


layer = 'inception_4c/3x3_reduce'

su.SetupCaffe.gpu_on()
net = ml.NetModels.setup_googlenet_model('../CommonCaffe/TrainedModels/')

style_data = gd.get_layers_data(net, 'ImagesIn/elephants2.jpg', layer)
subject_data = gd.get_layers_data(net, 'ImagesIn/smallerwonder.jpg', layer)

stl = dream_styles.Styles()

stl.setup_style_iterator(iterator[0])

for i in range(0, iterator[0]['iter_n']):
    vis = stl.next_frame(net, style_data, subject_data, layer)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    cv2.imshow('Video', vis)
    cv2.waitKey(1)

vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
display.Display().showResultPIL(vis)

