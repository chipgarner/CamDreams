# Motion detection based camera states

import cv2


class CamStates:

    state = ''
    motion = False
    start_time = 0
    dream_start = 0
    low_motion_last_time = 0
    LOW_MOTION_TIMEOUT = 240000000000
    LOW_THRESHOLD = 2000000
    HIGH_THRESHOLD = 6500000
    DREAM_OVER = 20000000000
    motion_threshold = HIGH_THRESHOLD

    def __init__(self):
        self.state = 'waiting'
        self.start_time = cv2.getTickCount() + 2000000000  # Startup delay
        self.dream_start = cv2.getTickCount()
        self.low_motion_last_time = cv2.getTickCount()
        pass

    def get_state(self, motion_detect):

        if self.state == 'start_dreaming':
            self.__dreaming_start()

        elif self.state == 'fade_dream_to_frame':
            self.state = 'fading'

        elif self.state == 'fading':
            self.__fading()

        elif motion_detect > self.motion_threshold:
            self.__on_motion_above_threshold()

        elif cv2.getTickCount() - self.low_motion_last_time > self.LOW_MOTION_TIMEOUT:
            self.state = 'waiting'

        else:
            if motion_detect > self.LOW_THRESHOLD:
                self.low_motion_last_time = cv2.getTickCount()

            if self.motion:
                self.start_time = cv2.getTickCount()
                self.motion = False
            else:
                if self.state == 'dreaming':
                    how_long = cv2.getTickCount() - self.dream_start
                    if how_long > self.DREAM_OVER:
                        self.state = 'fade_dream_to_frame'
                elif self.state == 'show_frames':
                    how_long = cv2.getTickCount() - self.start_time
                    if how_long > 7000000000:
                        self.state = 'start_dreaming'

        return self.state

    def __dreaming_start(self):
        self.motion_threshold = self.HIGH_THRESHOLD
        self.dream_start = cv2.getTickCount()
        self.state = 'dreaming'

    def __on_motion_above_threshold(self):
        self.motion = True
        self.low_motion_last_time = cv2.getTickCount()
        if self.state == 'dreaming':
            # self.motion_threshold = self.LOW_THRESHOLD
            self.state = 'show_frames'
        else:
            self.state = 'start_dreaming'

    beta = 0.0
    fade_iterations = 80.0
    fade_iter = 0.0

    def __fading(self):
        self.fade_iter += 1.0

        if self.fade_iter > self.fade_iterations:
            self.state = 'show_frames'
            self.fade_iter = 0.0
            self.beta = 0.0
        else:
            self.beta = self.fade_iter / self.fade_iterations
