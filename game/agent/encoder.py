import numpy as np
import cv2
from termcolor import colored

class StateActionEncoder:
  """
  Dummy state, action encoder.
  Please override!
  """

  def encode_state(self, s):
    return s, np.array2string(s, precision=0)

  def encode_action(self, s):
    return np.array2string(s, precision=2, separator=',')

  def decode_action(self, s):
    return np.fromstring(s.replace("[","").replace("]",""), sep=',')

  def encode_color(self, c):
    b,g,r = c
    if min(b,g,r)==0: return 0 # Black
    elif g>=200 or g>=2*r: return 255 # Green
    elif abs(100-r)<30 and abs(100-g)<30 and abs(100-b)<30: return 128 # Gray tile
    else: return 64 # Otherwise
 

class CarRaceEncoder(StateActionEncoder):

  def __init__(self):
    self.n = 0

  def is_gray(self,b,g,r):
    return max(b,g,r) - min(b,g,r) <= 30

  def encode_state(self, s):
    frame = s[10:90, 0:80, :] # 80x80

    ratio = 5 # Compress ratio
    tilesize = 80//ratio

    compressed = cv2.resize(frame, (tilesize,tilesize))

    vector = []

    box = np.zeros_like(frame)
    for j in range(tilesize):
      for i in range(tilesize):
        [b,g,r] = [int(k) for k in compressed[j,i]]
        if self.is_gray(b,g,r):
          vector.append(1)
          cv2.rectangle(box, (i*ratio,j*ratio), (i*ratio+ratio,j*ratio+ratio), [160,160,160], -1)
        else:
          vector.append(0)
          cv2.rectangle(box, (i*ratio,j*ratio), (i*ratio+ratio,j*ratio+ratio), [255,255,255], -1)

    vector = np.array(vector)
    if self.n<100:
      filename = "debug/f-{:4}.png".format(self.n)
      cv2.imwrite(filename, box)
      self.n = self.n+1

    return vector, np.array2string(vector, precision=0)    
