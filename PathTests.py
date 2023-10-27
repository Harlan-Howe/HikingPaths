import unittest
import cv2
import numpy as np

import PathMakerFile
from PathMakerFile import PathMaker, CostModeType

class MyTestCase(unittest.TestCase):

    # @unittest.skip("Activate this when you have written perform_search().")
    def test_1_display_path(self):
        pm = PathMaker(filename="40x40Gray.jpg")
        best_g_array = \
            [[999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47,
              46, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 58, 57, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 45, 44, 43, 42, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 61, 60, 59, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 41, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 62, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 40, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 64, 63, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 39, 38, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 66, 65, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 37, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 67, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 36, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 68, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 35, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 69, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 34, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 70, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 33, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 71, 999, 999, 999, 999, 999, 999, 999, 999, 999, 1, 0, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 32, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 72, 999, 999, 999, 999, 999, 999, 999, 999, 2, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 31, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 73, 999, 999, 999, 999, 999, 999, 999, 4, 3, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 30, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 74, 999, 999, 999, 999, 999, 999, 999, 5, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 29, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 75, 999, 999, 999, 999, 999, 999, 999, 6, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 28, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 76, 999, 999, 999, 999, 999, 999, 999, 7, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 27, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 77, 999, 999, 999, 999, 999, 999, 8, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 24, 25, 26, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 78, 999, 999, 999, 999, 999, 999, 9, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 22, 23, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 10, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 20, 21, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 11, 12, 13, 14, 15, 16, 17, 18,
              19, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
             [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
              999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999]]
        pm.best_g = np.asarray(best_g_array)
        previous_point_array = \
            [[[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19], [11, 20],
              [11, 21], [11, 22], [11, 23], [12, 24], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [12, 12], [11, 13], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [12, 25], [12, 26], [12, 27], [13, 28], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [13, 9], [13, 10],
              [12, 11], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [14, 29], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [13, 8], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [15, 29],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [15, 8], [14, 8], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [15, 30],
              [16, 30], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [16, 7], [15, 7], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [17, 30], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [16, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [18, 30], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [17, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [19, 30], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [18, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [20, 30], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [19, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [21, 30], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [20, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [21, 17], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [22, 30], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [21, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [21, 16], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [23, 30], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [22, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [23, 15], [22, 15], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [24, 30], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [23, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [23, 14], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [25, 29], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [24, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [24, 14], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [26, 28],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [25, 6], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [25, 14], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [27, 28], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [26, 6], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [26, 14], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [28, 26], [27, 26], [27, 27], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [27, 7], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [27, 14], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [29, 25], [28, 25], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [28, 14], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [30, 23], [29, 24], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [29, 14], [30, 15], [30, 16], [30, 17], [30, 18],
              [30, 19], [30, 20], [30, 21], [30, 22], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
             [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
              [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]]

        pm.previous_point = np.asarray(previous_point_array)
        pm.start_point_x_y = (17, 21)
        pm.start_point_r_c = (21, 17)
        pm.end_point_x_y = (7, 28)
        pm.end_point_r_c = (28, 7)
        pm.display_path(pm.end_point_r_c)
        pm.show_map()
        print("I've just displayed a window. It should show a spiral -- see 'expectedSpiral.png' in this project. "
              "Click in it and press any key to 'pass' this test.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        self.assertEqual(True, True)  # add assertion here

    # @unittest.skip("Activate this when you have written perform_search().")
    def test_2_find_straight_path(self):
        pm = PathMaker(filename="Small_picture.jpg")
        PathMakerFile.COST_MODE = CostModeType.HIGH_EXPENSIVE
        pm.start_point_x_y = (10, 20)
        pm.start_point_r_c = (20, 10)
        pm.end_point_x_y = (60, 55)
        pm.end_point_r_c = (55, 60)
        PathMakerFile.ALPHA = 0.0
        path = pm.perform_search()
        if path is not None:
            print(f"Found path with length {len(path)}.")
        else:
            print("No path found.")
        pm.reset()
        pm.draw_start_point()
        pm.draw_end_point()
        pm.display_path(pm.end_point_r_c, (255,255,0))
        pm.show_map()
        print("I've just displayed a window. It should show a cyan direct line from start to finish made of a diagonal and/or"
              "one horizontal or vertical line."
              "Click in it and press any key to 'pass' this test.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # @unittest.skip("Activate this when you have written perform_search().")
    def test_3_find_river_path(self):
        pm = PathMaker(filename="Small_picture.jpg")
        PathMakerFile.COST_MODE = CostModeType.HIGH_EXPENSIVE
        pm.start_point_x_y = (12, 35)
        pm.start_point_r_c = (35, 12)
        pm.end_point_x_y = (60, 15)
        pm.end_point_r_c = (15, 60)
        PathMakerFile.ALPHA = 50
        path = pm.perform_search()
        if path is not None:
            print(f"Found path with length {len(path)}.")
        else:
            print("No path found.")
        pm.reset()
        pm.draw_start_point()
        pm.draw_end_point()
        pm.display_path(pm.end_point_r_c, (0, 255, 255))
        pm.show_map()
        print("I've just displayed a window. It should show a yellow line from start to finish that follows the dark parts"
              "of the image."
              "Click in it and press any key to 'pass' this test.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # @unittest.skip("Activate this when you have written perform_search() and are satisfied with tests 2 & 3..")
    def test_4_big_test(self):
        pm = PathMaker(filename="new_england height map.jpg")
        PathMakerFile.COST_MODE = CostModeType.HIGH_EXPENSIVE
        pm.start_point_x_y = (350, 5)
        pm.start_point_r_c = (5, 350)
        pm.end_point_x_y = (20, 340)
        pm.end_point_r_c = (340, 20)
        PathMakerFile.ALPHA = 50
        endpoint = pm.perform_search()
        if endpoint is not None:
            print(f"Found path.")
        else:
            print("No path found.")
        pm.reset()
        pm.draw_start_point()
        pm.draw_end_point()
        pm.display_path(pm.end_point_r_c, (0, 255, 255))
        pm.show_map()
        print(
            "I've just displayed a window. It should show the new england graph, with a path that goes down the Hudson."
            "Click in it and press any key to 'pass' this test.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
