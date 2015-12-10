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

    assert ret == 'show_frames'
    assert cv2.getTickCount() - cs.start_time < 5000
    assert not cs.motion

  def test_get_state_low_motion_false_long_time(self):
    cs = cam_states.CamStates()
    cs.motion = False
    cs.start_time = 0

    ret = cs.get_state(cs.LOW_THRESHOLD - 1)

    assert ret == 'start_dreaming'
    assert not cs.motion

  def test_get_state_starts_frames(self):
    cs = cam_states.CamStates()

    cs.get_state(cs.LOW_THRESHOLD)
    cs.get_state(cs.LOW_THRESHOLD)
    cs.get_state(cs.LOW_THRESHOLD)
    ret = cs.get_state(cs.LOW_THRESHOLD)

    assert ret == 'show_frames'
    assert not cs.motion

  def test_get_state_start_dreaming_high_threshold(self):
    cs = cam_states.CamStates()
    cs.motion = False
    cs.start_time = 0

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

  def test_get_state_restart_dream_timeout(self):
    cs = cam_states.CamStates()
    cs.motion = False
    cs.state = 'dreaming'
    cs.dream_start = cv2.getTickCount() - cs.DREAM_OVER - 1

    ret = cs.get_state(cs.LOW_THRESHOLD - 1)

    assert ret == 'start_dreaming'