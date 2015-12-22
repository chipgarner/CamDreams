from unittest import TestCase
import cam_states_faces
import cv2


class TestCamStatesFaces(TestCase):
    def test_get_state_waiting_starts_timer(self):
        cs = cam_states_faces.CamStatesFaces()

        state = cs.get_state(True)

        assert state == 'waiting'
        assert cv2.getTickCount() - cs.faces_latest_time < 5000
        assert cs.faces

    def test_get_state_to_dreaming_on_timeup(self):
        cs = cam_states_faces.CamStatesFaces()

        cs.get_state(True)
        cs.start_time -= cs.FACES_DELAY_TIMER
        state = cs.get_state(True)

        assert state == 'start_dreaming'

    def test_get_state_fade_to_dream(self):
        cs = cam_states_faces.CamStatesFaces()

        cs.state = 'dreaming'
        cs.dream_start = cv2.getTickCount() - cs.DREAM_OVER
        state = cs.get_state(True)

        assert state == 'fade_dream_to_frame'

    def test_get_state_to_waiting(self):
        cs = cam_states_faces.CamStatesFaces()

        cs.state = 'show_frames'
        cs.faces_latest_time -= cs.NO_FACES_TIMEOUT
        cs.faces = False
        state = cs.get_state(False)

        assert state == 'waiting'

    def test_get_fade_backgrounds_on_dream_count(self):
        cs = cam_states_faces.CamStatesFaces()

        cs.state = 'fading'
        cs.dream_count = 4
        cs.fade_iter = cs.fade_iterations
        state = cs.get_state(False)

        assert state == 'fade_backgrounds'
        assert cs.dream_count == 0

    def test_get_fading_backgrounds(self):
        cs = cam_states_faces.CamStatesFaces()

        cs.state = 'fade_backgrounds'
        state = cs.get_state(False)

        assert state == 'fading_backgrounds'

    def test_get_fading_backgrounds_increments(self):
        cs = cam_states_faces.CamStatesFaces()

        cs.state = 'fading_backgrounds'
        cs.fade_iter = 10
        state = cs.get_state(False)

        assert state == 'fading_backgrounds'
        assert cs.fade_iter == 11

    def test_get_fade_backgrounds_on_dream_count_continue(self):
        cs = cam_states_faces.CamStatesFaces()

        cs.state = 'fading'
        cs.dream_count = 4
        cs.fade_iter = cs.fade_iterations
        state = cs.get_state(False)

        assert state == 'fade_backgrounds'
        assert cs.dream_count == 0

        cs.state = 'fading'
        cs.fade_iter = cs.fade_iterations
        state = cs.get_state(False)

        assert state == 'show_frames'
        assert cs.dream_count == 1

        cs.state = 'fading'
        cs.fade_iter = cs.fade_iterations
        state = cs.get_state(False)

        assert state == 'show_frames'
        assert cs.dream_count == 2

        cs.state = 'fading'
        cs.fade_iter = cs.fade_iterations
        state = cs.get_state(False)

        assert state == 'show_frames'
        assert cs.dream_count == 3

        cs.state = 'fading'
        cs.fade_iter = cs.fade_iterations
        state = cs.get_state(False)

        assert state == 'show_frames'
        assert cs.dream_count == 4

        cs.state = 'fading'
        cs.fade_iter = cs.fade_iterations
        state = cs.get_state(False)

        assert state == 'fade_backgrounds'
        assert cs.dream_count == 0

