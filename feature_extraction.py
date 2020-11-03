import numpy as np
import cv2

def delauney_triangulation(landmarks):
    # find the space we want to partition, in this case the image size
    rectangle = cv2.boundingRect(cv2.convexHull(landmarks))

    # create a subdivision for triangulation
    subdivision = cv2.Subdiv2D(rectangle)
    
    # populate the subdivision with our facial landmark points
    for point in landmarks:
        subdivision.insert(tuple(point))

    delauney_triangles = []

    # create triangles from the points in the subdivision
    for triangle in subdivision.getTriangleList():

        # get the vertices of triangle
        v1 = (int(triangle[0]), int(triangle[1]))
        v2 = (int(triangle[2]), int(triangle[3]))
        v3 = (int(triangle[4]), int(triangle[5]))

        idx1 = landmarks.tolist().index(list(v1))
        idx2 = landmarks.tolist().index(list(v2))
        idx3 = landmarks.tolist().index(list(v3))

        # store the landmark point indices that make up each triangle
        triangle = [idx1, idx2, idx3]
        delauney_triangles.append(triangle)

    return delauney_triangles