from unittest import TestCase
import cv2
import cam_states


class TestCamStates(TestCase):
  def test_get_state_high_motion_set_true(self):
    cs = cam_states.CamStates()
    cs.state = 'dreaming'

    ret = cs.get_state(5000000)

    assert ret == 'show_frames'
    assert cs.motion

  def test_get_state_low_motion_true_starts_time(self):
    cs = cam_states.CamStates()
    cs.motion = True

    ret = cs.get_state(3000000)

    assert ret == 'show_frames'
    assert cv2.getTickCount() - cs.start_time < 5000
    assert not cs.motion

  def test_get_state_low_motion_false_long_time(self):
    cs = cam_states.CamStates()
    cs.motion = False
    cs.start_time = 0

    ret = cs.get_state(3000000)

    assert ret == 'start_dreaming'
    assert not cs.motion

