import cv2
import cam_states_faces
import numpy as np
import dreamer
import images
import face_detector


class CamDreams:
    def __init__(self):
        self.dr = dreamer.Dreamer()

        # For a full screen window
        self.video_capture = cv2.VideoCapture(0)
        self.cs = cam_states_faces.CamStatesFaces()

        self.backgrounds = ['Paintings/imagine.jpg',
                            'Paintings/flower.jpg',
                            'Paintings/BigSue53.jpg',
                            'Paintings/figures.jpg',
                            'Paintings/flower2.jpg',
                            'Paintings/lazy.jpg',
                            'Paintings/flower3.jpg',
                            'Paintings/floating2.jpg',
                            'Paintings/wondering.jpg',
                            'Paintings/seduction.jpg',
                            'Paintings/spring2.jpg',
                            'Paintings/rouge.jpg']
        self.background = cv2.imread(self.backgrounds[0])
        self.background = images.Images.resize_image(480, 640, self.background)

        cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    dr = None
    video_capture = None
    cs = None
    last_frame = None
    backgrounds = None
    back_index = 0
    background = None
    old_background = None
    dream_image = None

    # This is to make the edges black on my 1920 X 1200 (16:10 vs 4:3 camera
    # ) display
    black_edges = np.zeros((480, 768, 3), np.uint8)

    def __show_image(self, image):
        self.black_edges[0:480, 64:704] = image
        cv2.imshow('Video', self.black_edges)

    def __fade(self, start_image, end_image):
        beta = self.cs.beta
        alpha = 1.0 - beta

        return cv2.addWeighted(start_image, alpha, end_image, beta, 0.0)

    # empty out the frame buffer and get the latest frame
    def __get_next_frame(self):
        delta_t = 0.0
        while delta_t < 30000000.0:
            t = cv2.getTickCount()
            self.video_capture.grab()
            delta_t = cv2.getTickCount() - t
        rt, frm = self.video_capture.retrieve()
        self.last_frame = frm
        return frm

    def __load_next_background(self):
        self.old_background = self.background.copy()
        self.back_index += 1
        if self.back_index >= len(self.backgrounds):
            self.back_index = 0
        self.background = cv2.imread(self.backgrounds[self.back_index])

        # resize to camera frame size
        self.background = images.Images.resize_image(480, 640, self.background)

    def __do_state(self, state, frame):

        if state == 'show_frames':
            self.__show_image(frame)
        elif state == 'waiting':
            self.__show_image(frame)
        elif state == 'fade_dream_to_frame':
            self.dream_image = self.dr.get_dream_frame()
            frame = self.__fade(self.dream_image, frame)
            self.__show_image(frame)
        elif state == 'fading':
            frame = self.__fade(self.dream_image, frame)
            self.__show_image(frame)
        elif state == 'fade_backgrounds':
            self.__load_next_background()
            frame = self.__fade(self.old_background, self.background)
            self.__show_image(frame)
        elif state == 'fading_backgrounds':
            frame = self.__fade(self.old_background, frame)
            self.__show_image(frame)
        elif state == 'start_dreaming':
            self.dr.start_dreaming(frame)
        elif state == 'dreaming':
            self.dream_image = self.dr.get_dream_frame()
            self.__show_image(self.dream_image)
        else:
            print state + ' state not found error.'
            self.__show_image(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

    def run(self):

        ret, frame = self.video_capture.read()  # Initialize the last_frame
        self.last_frame = frame
        fc = face_detector.FaceDetector()

        while True:

            frame = self.__get_next_frame()

            got_face, frame = fc.get_faces(frame, self.background)

            state = self.cs.get_state(got_face)
            self.__do_state(state, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        self.video_capture.release()
        cv2.destroyAllWindows()


CamDreams().run()
