import cv2, numpy as np
import random
from scipy import spatial

def triangulation(target_frame, hull2):
    # Find delanauy traingulation for convex hull points
    sizeImg2 = target_frame.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles_spatial(rect, hull2, target_frame)

    return dt

def calculateDelaunayTriangles_spatial(rect, points, img):
	# visualizeDelaunay(rect, points, img)
	Tri = spatial.Delaunay(points)
	triangles = Tri.simplices

	return listOfListToTuples(triangles.tolist())

def listOfListToTuples(p):
	t_list = []
	for ent in p:
		s = []
		for i in range(0, len(ent)):
			s.append(ent[i])
		t_list.append(tuple(s))

	return t_list