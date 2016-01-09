# Face detection based camera states

import cv2


class CamStatesFaces:

    state = ''
    faces = False
    start_time = 0
    dream_start = 0
    faces_latest_time = 0
    dream_count = 1
    NO_FACES_TIMEOUT = 240000000000
    PERSIST_FACES_TIMER = 5000000000
    FACES_DELAY_TIMER = 20000000000
    DREAM_OVER = 25000000000

    def __init__(self):
        self.state = 'waiting'
        self.start_time = cv2.getTickCount() + 2000000000  # Startup delay
        self.dream_start = cv2.getTickCount()
        self.faces_latest_time = cv2.getTickCount()
        pass

    def get_state(self, face_present):

        if self.state == 'start_dreaming':
            self.__dreaming_start()

        elif self.state == 'dreaming':
            how_long = cv2.getTickCount() - self.dream_start
            if how_long > self.DREAM_OVER:
                self.state = 'fade_dream_to_frame'

        elif self.state == 'fade_dream_to_frame':
            self.state = 'fading'

        elif self.state == 'fade_backgrounds':
            self.state = 'fading_backgrounds'

        elif self.state == 'fading_backgrounds':
            self.__fading()

        elif self.state == 'fading':
            self.__fading()

        elif face_present:
            self.__on_faces_present()

        elif self.faces:
            if cv2.getTickCount() - self.faces_latest_time > self.PERSIST_FACES_TIMER:
                self.faces = False

        elif cv2.getTickCount() - self.faces_latest_time > self.NO_FACES_TIMEOUT:
            self.state = 'waiting'
        else:
            if self.state != 'waiting':
                self.state = 'start_dreaming'

        return self.state

    def __dreaming_start(self):
        self.dream_start = cv2.getTickCount()
        self.state = 'dreaming'

    def __on_faces_present(self):
        if not self.faces:
            if self.state == 'waiting':
                self.start_time = cv2.getTickCount()
        elif cv2.getTickCount() - self.start_time > self.FACES_DELAY_TIMER:
            if self.state != 'dreaming':
                self.state = 'start_dreaming'

        self.faces = True
        self.faces_latest_time = cv2.getTickCount()

    beta = 0.0
    fade_iterations = 80.0
    fade_iter = 0.0

    def __fading(self):
        self.fade_iter += 1.0

        if self.fade_iter > self.fade_iterations:
            self.state = 'show_frames'
            if not self.state == 'fading_backgrounds':
                self.dream_count += 1
                if self.dream_count > 4:
                    self.dream_count = 0
                    self.state = 'fade_backgrounds'

            self.fade_iter = 0.0
            self.beta = 0.0
        else:
            self.beta = self.fade_iter / self.fade_iterations
