from unittest import TestCase
import cv2
import cam_states


class TestCamStates(TestCase):
    def test_get_state_high_motion_set_true(self):
        cs = cam_states.CamStates()
        cs.state = 'dreaming'

        ret = cs.get_state(cs.HIGH_THRESHOLD + 1)

        assert ret == 'show_frames'
        assert cs.motion

    def test_get_state_low_motion_true_starts_time(self):
        cs = cam_states.CamStates()
        cs.motion = True

        ret = cs.get_state(cs.LOW_THRESHOLD - 1)

        assert ret == 'waiting'
        assert cv2.getTickCount() - cs.start_time < 5000
        assert not cs.motion

    def test_get_state_low_motion_false_long_time(self):
        cs = cam_states.CamStates()
        cs.motion = False
        cs.start_time = 0
        cs.state = 'not waiting'

        ret = cs.get_state(cs.LOW_THRESHOLD - 1)

        assert ret == 'start_dreaming'
        assert not cs.motion

    def test_get_state_starts_waiting(self):
        cs = cam_states.CamStates()

        cs.get_state(cs.LOW_THRESHOLD)
        cs.get_state(cs.LOW_THRESHOLD)
        cs.get_state(cs.LOW_THRESHOLD)
        ret = cs.get_state(cs.LOW_THRESHOLD)

        assert ret == 'waiting'
        assert not cs.motion

    def test_get_state_start_dreaming_high_threshold(self):
        cs = cam_states.CamStates()
        cs.motion = False
        cs.start_time = 0
        cs.state = 'not waiting'

        cs.get_state(cs.LOW_THRESHOLD - 1)
        ret = cs.get_state(cs.LOW_THRESHOLD - 1)

        assert cs.motion_threshold == cs.HIGH_THRESHOLD
        assert ret == 'dreaming'
        assert not cs.motion

    def test_get_state_motion_high_to_low_threshold(self):
        cs = cam_states.CamStates()
        cs.motion = False
        cs.state = 'dreaming'
        cs.motion_threshold = cs.HIGH_THRESHOLD

        cs.get_state(cs.HIGH_THRESHOLD + 1)

        assert cs.motion_threshold == cs.LOW_THRESHOLD

    def test_get_state_dream_over_fade_to_frame_and_start_dreaming(self):
        cs = cam_states.CamStates()
        cs.motion = False
        cs.state = 'dreaming'
        cs.dream_start = cv2.getTickCount() - cs.DREAM_OVER - 1
        cs.start_time = cv2.getTickCount() - 500000000 - 1

        ret = cs.get_state(cs.LOW_THRESHOLD - 1)

        assert ret == 'fade_dream_to_frame'

        ret = cs.get_state(cs.LOW_THRESHOLD - 1)

        assert ret == 'start_dreaming'

    def test_get_state_start_up_nothing_until_high_motion(self):
        cs = cam_states.CamStates()

        ret = cs.get_state(cs.LOW_THRESHOLD + 1)

        assert ret == 'waiting'
        assert cs.motion_threshold == cs.HIGH_THRESHOLD
        assert not cs.motion

    def test_get_state_waiting_at_startup(selfself):
        cs = cam_states.CamStates()

        ret = cs.get_state(cs.HIGH_THRESHOLD - 1)

        assert not cs.motion
        assert ret == 'waiting'
