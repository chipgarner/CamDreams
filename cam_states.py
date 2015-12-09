import cv2


class CamStates:

    state = ''
    t = 0
    motion = False
    start_time = cv2.getTickCount()

    def __init__(self):
        self.state = 'show_frames'
        self.t = cv2.getTickCount()
        pass

    def get_state(self, motion_detect):

        if self.state == 'start_dreaming':
            self.state = 'dreaming'
            return self.state

        if motion_detect > 4000000:
            self.motion = True
            if self.state == 'dreaming':
                self.state = 'show_frames'

        else:
            if self.motion:
                self.start_time = cv2.getTickCount()
                self.motion = False
            else:
                how_long = cv2.getTickCount() - self.start_time
                if how_long > 2000000000:
                    if self.state != 'dreaming':
                        self.state = 'start_dreaming'
                        # subject_data = gd.get_layers_data_image(net, frame, layer)
                        # stl.setup_style_iterator(iterator[0])
                        # dreaming = True

        return self.state
