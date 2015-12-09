import cv2
import dream_styles
import setup_caffe_network as su
import models as ml
import get_layer_data as gd

iterator = [
    {
        'iter_n': 10,
        'start_sigma': 6.0,
        'end_sigma': 0.0,
        'start_step_size': 6.0,
        'end_step_size': 6.0
    },

]


layer = 'inception_4c/3x3_reduce'

su.SetupCaffe.gpu_on()
net = ml.NetModels.setup_googlenet_model('../CommonCaffe/TrainedModels/')

style_data = gd.get_layers_data(net, 'ImagesIn/elephants2.jpg', layer)
stl = dream_styles.Styles()

video_capture = cv2.VideoCapture(0)
# video_capture.set(3, 1280)
# video_capture.set(4, 1024)
cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

last_frame = None
dreaming = False

start_time = cv2.getTickCount()
how_long = 0
motion = False

while True:

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    if last_frame is None:
        last_frame = gray

    diff = cv2.absdiff(gray, last_frame)
    sum_diff = cv2.sumElems(diff)
    last_frame = gray

    if dreaming:
        vis = stl.next_frame(net, style_data, subject_data, layer)
        cv2.imshow('Video', vis)
    else:
        cv2.imshow('Video', frame)

    if sum_diff[0] > 4000000:
        motion = True
        if dreaming:
            dreaming = False

    else:
        if motion:
            start_time = cv2.getTickCount()
            motion = False
        else:
            how_long = cv2.getTickCount() - start_time
            if how_long > 2000000000:
                motion = False
                if not dreaming:
                    subject_data = gd.get_layers_data_image(net, frame, layer)
                    stl.setup_style_iterator(iterator[0])
                    dreaming = True

    # if sum_diff[0] > 2000000:
    #     if motion:
    #         how_long = cv2.getTickCount() - start_time
    #
    #         if how_long > 2000000000:
    #             motion = False
    #             if dreaming:
    #                 dreaming = False
    #             else:
    #                 subject_data = gd.get_layers_data_image(net, frame, layer)
    #                 stl.setup_style_iterator(iterator)
    #                 dreaming = True
    #     else:
    #         motion = True
    #         start_time = cv2.getTickCount()
    # else:
    #     if motion:
    #         how_long = cv2.getTickCount() - start_time
    #         if how_long > 2000000000:
    #             motion = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if dreaming:
            break
        else:
            dreaming = True

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
