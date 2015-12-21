from unittest import TestCase
import cam_states_faces
import cv2


class TestCamStatesFaces(TestCase):
    def test_get_state_starts_timer(self):
        cs = cam_states_faces.CamStatesFaces()

        state = cs.get_state(True)

        assert state == 'waiting'
        assert cv2.getTickCount() - cs.faces_latest_time < 5000
        assert cs.faces
