import numpy as np
import cv2
from constants.constants import debug_convex_hull
from constants.constants import debug_delauney_triangulation

# need to get points from dlib

hullIndex = cv2.convexHull(points, returnPoints = False)

