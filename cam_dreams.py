import cv2
import dream_styles
import setup_caffe_network as su
import models as ml
import get_layer_data as gd
import cam_states
import numpy as np

iterator = [
    {
        'iter_n': 30,
        'start_sigma': 2.0,
        'end_sigma': 0.0,
        'start_step_size': 6.0,
        'end_step_size': 6.0
    },

]


layers = ['inception_3b/5x5_reduce', 'inception_4a/5x5_reduce', 'inception_4c/3x3_reduce', 'inception_5a/3x3_reduce']
i_layer = 0

su.SetupCaffe.gpu_on()
net = None

stl = dream_styles.Styles()

video_capture = cv2.VideoCapture(0)
cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

cs = cam_states.CamStates()

last_frame = None

# This is to make the edges black on my 1920 X 1200 (16:10 vs 4:3) display
height = 480
width = 768
black_edges = np.zeros((height, width, 3), np.uint8)


def show_image(image):
    black_edges[0:480, 64:704] = image
    cv2.imshow('Video', black_edges)


while True:

    ret, frame = video_capture.read()

    if last_frame is None:
        last_frame = frame

    diff = cv2.absdiff(frame, last_frame)
    sum_diff = cv2.sumElems(diff)
    last_frame = frame

    state = cs.get_state(sum_diff[0])
    if state == 'show_frames':
        show_image(frame)
    elif state == 'start_dreaming':
        layer = layers[i_layer]
        if i_layer == 0:  # Results change if not reloaded, there may be something in the network tha just need reseting
            net = ml.NetModels.setup_googlenet_model('../CommonCaffe/TrainedModels/')
        i_layer += 1
        if i_layer >= len(layers):
            i_layer = 0
        print layer
        style_data = gd.get_layers_data(net, 'ImagesIn/425 barn.jpg', layer)
        subject_data = gd.get_layers_data_image(net, frame, layer)
        stl.setup_style_iterator(iterator[0])
        vis = stl.next_frame(net, style_data, subject_data, layer)
        show_image(vis)
    elif state == 'dreaming':
        vis = stl.next_frame(net, style_data, subject_data, layer)
        show_image(vis + diff / 3)
        for i in range(0, 3):  # This is slow so empty the video buffer
            video_capture.grab()
    else:
        show_image(diff * 5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
