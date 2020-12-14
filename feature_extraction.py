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


def draw_delauney_triangles(triangles, landmarks, img):
    for i, j, k in triangles:

        v1 = tuple(landmarks[i])
        v2 = tuple(landmarks[j])
        v3 = tuple(landmarks[k])

        cv2.line(img, v1, v2, (255, 255, 255), 1, cv2.LINE_AA, 0)
        cv2.line(img, v2, v3, (255, 255, 255), 1, cv2.LINE_AA, 0)
        cv2.line(img, v3, v1, (255, 255, 255), 1, cv2.LINE_AA, 0)

    cv2.imshow("Delauney Triangulation", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_convex_hull(points, img):
    for point in points:
        x = point[0][0]
        y = point[0][1]
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Convex Hull", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convex_hull(points1, points2):
    hullIndex = cv2.convexHull(np.array(points2).astype(np.int32), returnPoints=False)

    hull1 = [points1[int(hullIndex[i])] for i in range(0,len(hullIndex))]
    hull2 = [points2[int(hullIndex[i])] for i in range(0, len(hullIndex))]

    return hull1, hull2