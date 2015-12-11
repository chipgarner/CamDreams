import cv2


class CamStates:

    state = ''
    motion = False
    start_time = 0
    dream_start = 0
    LOW_THRESHOLD = 2000000
    HIGH_THRESHOLD = 4500000
    DREAM_OVER = 20000000000
    motion_threshold = HIGH_THRESHOLD

    def __init__(self):
        self.state = 'waiting'
        self.start_time = cv2.getTickCount() + 2000000000  # Startup delay
        self.dream_start = cv2.getTickCount()
        pass

    def get_state(self, motion_detect):

        if self.state == 'start_dreaming':
            self.__dreaming_start()

        elif self.state == 'fade_dream_to_frame':
            self.state = 'fading'

        elif motion_detect > self.motion_threshold:
            self.__on_motion_above_threshold()
        else:
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
                    if how_long > 500000000:
                        self.state = 'start_dreaming'

        return self.state

    def __dreaming_start(self):
        self.motion_threshold = self.HIGH_THRESHOLD
        self.dream_start = cv2.getTickCount()
        self.state = 'dreaming'

    def __on_motion_above_threshold(self):
        self.motion = True
        if self.state == 'dreaming':
            # self.motion_threshold = self.LOW_THRESHOLD
            self.state = 'show_frames'
        elif self.state == 'waiting':
            self.state = 'start_dreaming'
