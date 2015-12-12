import cv2
import cam_states
import numpy as np
import dreamer


class CamDreams:

    dr = dreamer.Dreamer()

    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    cs = cam_states.CamStates()

    last_frame = None

    # This is to make the edges black on my 1920 X 1200 (16:10 vs 4:3) display
    height = 480
    width = 768
    black_edges = np.zeros((height, width, 3), np.uint8)

    def __show_image(self, image):
        self.black_edges[0:480, 64:704] = image
        cv2.imshow('Video', self.black_edges)

    iterations = 80.00
    index = 0.0

    def __fade(self, start_image, end_image):
        beta = self.cs.beta
        alpha = 1.0 - beta

        return cv2.addWeighted(start_image, alpha, end_image, beta, 0.0)

    # empty put the frame buffer
    def __get_next_frame_and_diff(self):
        delta_t = 0.0
        while delta_t < 30000000.0:
            t = cv2.getTickCount()
            self.video_capture.grab()
            delta_t = cv2.getTickCount() - t
        rt, frm = self.video_capture.retrieve()
        dif = cv2.absdiff(frm, self.last_frame)
        self.last_frame = frm
        return frm, dif

    def run(self):

        ret, frame = self.video_capture.read()  # Initialize the last_frame
        self.last_frame = frame

        for num_frames in range(0, 10):  # The first few frames can be noisy
            frame, diff = self.__get_next_frame_and_diff()
            self.__show_image(diff * 5)

        while True:

            frame, diff = self.__get_next_frame_and_diff()

            sum_diff = cv2.sumElems(diff)

            state = self.cs.get_state(sum_diff[0])
            if state == 'show_frames':
                self.__show_image(frame)
            elif state == 'waiting':
                self.__show_image(diff * 5)
            elif state == 'fade_dream_to_frame':
                vis = self.dr.get_dream_frame()
                frame = self.__fade(vis, frame)
                self.__show_image(frame)
            elif state == 'fading':
                frame = self.__fade(vis, frame)
                self.__show_image(frame)
            elif state == 'start_dreaming':
                vis = self.dr.start_dreaming(frame)
            elif state == 'dreaming':
                vis = self.dr.get_dream_frame()
                self.__show_image(vis + diff / 3)
            elif state == 'dark_screen':
                self.__show_image(diff + 5)
            else:
                print state + ' state not found error.'
                self.__show_image(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        self.video_capture.release()
        cv2.destroyAllWindows()

CamDreams().run()
