import numpy as np

def get_sticks():
  stick_0 = np.zeros((7,7))
  stick_0[:4,3:] = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 1]
  ])
  stick_1 = np.zeros((7,7))
  stick_1[:4,3:] = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0]
  ])
  stick_2 = np.zeros((7,7))
  stick_2[:4,3:] = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 0]
  ])
  stick_3 = np.zeros((7,7))
  stick_3[:4,3:] = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0]
  ])
  stick_4 = np.zeros((7,7))
  stick_4[:4,3:] = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0]
  ])
  stick_5 = np.zeros((7,7))
  stick_5[:4,3:] = np.array([
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0]
  ])
  stick_full = np.zeros((24,7,7))
  for i in range(0,4):
    j = i*6
    stick_full[j+0] = stick_0
    stick_full[j+1] = stick_1
    stick_full[j+2] = stick_2
    stick_full[j+3] = stick_3
    stick_full[j+4] = stick_4
    stick_full[j+5] = stick_5
    stick_0 = np.rot90(stick_0)
    stick_1 = np.rot90(stick_1)
    stick_2 = np.rot90(stick_2)
    stick_3 = np.rot90(stick_3)
    stick_4 = np.rot90(stick_4)
    stick_5 = np.rot90(stick_5)
  return stick_full