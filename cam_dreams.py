import cv2
import dream_styles
import setup_caffe_network as su
import models as ml
import get_layer_data as gd
import cam_states
import numpy as np


class CamDreams:
    iterator = [
        {
            'iter_n': 20,
            'start_sigma': 1.5,
            'end_sigma': 0.0,
            'start_step_size': 6.0,
            'end_step_size': 6.0
        },

    ]

    su.SetupCaffe.gpu_on()
    net = ml.NetModels.setup_googlenet_model('../CommonCaffe/TrainedModels/')

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

    def show_image(self, image):
        self.black_edges[0:480, 64:704] = image
        cv2.imshow('Video', self.black_edges)

    @staticmethod
    def input_filter(img):
        img = cv2.medianBlur(img, 3)
        img = cv2.medianBlur(img, 3)
        img = cv2.medianBlur(img, 3)
        img = cv2.medianBlur(img, 5)
        img = cv2.bilateralFilter(img, 20, 50, 10)
        return img

    iterations = 80.00
    index = 0.0

    def fade(self, start_image, end_image):

        alpha = 1.0 - self.index / self.iterations
        beta = self.index / self.iterations
        self.index += 1.0
        if self.index > self.iterations:
            self.index = 0
            self.cs.state = 'show_frames'  # HACK!

        return cv2.addWeighted(start_image, alpha, end_image, beta, 0.0)

    # empty put the frame buffer
    def __get_next_frame(self):
        delta_t = 0.0
        while delta_t < 30000000.0:
            t = cv2.getTickCount()
            self.video_capture.grab()
            delta_t = cv2.getTickCount() - t
        return self.video_capture.retrieve()

    def run(self):
        layers = ['inception_3b/5x5_reduce', 'inception_4a/5x5_reduce', 'inception_4c/3x3_reduce', 'inception_5a/3x3_reduce']
        i_layer = 0

        num_frames = 0

        while True:

            ret, frame = self.__get_next_frame()

            if self.last_frame is None:
                self.last_frame = frame

            diff = cv2.absdiff(frame, self.last_frame)
            sum_diff = cv2.sumElems(diff)
            self.last_frame = frame

            if num_frames < 10:  # The first few frames can be noisy
                num_frames += 1
                self.show_image(diff * 5)
                continue

            state = self.cs.get_state(sum_diff[0])
            print sum_diff[0]
            if state == 'show_frames':
                self.show_image(frame)
            elif state == 'waiting':
                self.show_image(diff * 5)
            elif state == 'fade_dream_to_frame':
                vis = self.stl.next_frame(self.net, style_data, subject_data, layer)
                frame = self.fade(vis, frame)
                self.show_image(frame)
            elif state == 'fading':
                frame = self.fade(vis, frame)
                self.show_image(frame)
            elif state == 'start_dreaming':
                layer = layers[i_layer]
                # if i_layer == 0:  # Results change if not reloaded, there may be something in the network tha just need reseting
                    # net = ml.NetModels.setup_googlenet_model('../CommonCaffe/TrainedModels/')
                i_layer += 1
                if i_layer >= len(layers):
                    i_layer = 0
                print layer
                style_data = gd.get_layers_data(self.net, 'ImagesIn/717 pickup.jpg', layer)
                subject_data = gd.get_layers_data_image(self.net, self.input_filter(frame), layer)
                self.stl.setup_style_iterator(self.iterator[0])
                vis = self.stl.next_frame(self.net, style_data, subject_data, layer)
                self.show_image(vis)
            elif state == 'dreaming':
                vis = self.stl.next_frame(self.net, style_data, subject_data, layer)
                self.show_image(vis + diff / 3)
            elif state == 'dark_screen':
                self.show_image(diff + 5)
            else:
                print state
                self.show_image(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        self.video_capture.release()
        cv2.destroyAllWindows()

CamDreams().run()