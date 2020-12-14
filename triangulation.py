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

def visualizeDelaunay(rect, points, img):
	subdiv = cv2.Subdiv2D(rect)
	for p in points:
		subdiv.insert(p)

	imgToShow = np.copy(img)
	draw_delaunay(imgToShow, subdiv)
	showBGRimage(imgToShow)

def calculateDelaunayTriangles_subdiv(rect, points, img):
	subdiv = cv2.Subdiv2D(rect)
	for p in points:
		subdiv.insert(p)

	# imgToShow = np.copy(img)
	# draw_delaunay(imgToShow, subdiv)
	# showBGRimage(imgToShow)

	triangleList = subdiv.getTriangleList()
	delaunayTri = []
	pt = []

	for t in triangleList:
		pt.append((t[0], t[1]))
		pt.append((t[2], t[3]))
		pt.append((t[4], t[5]))

		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])

		if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
			ind = []
			for j in xrange(0, 3):
				for k in xrange(0, len(points)):
					if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
						ind.append(k)
			if len(ind)==3:
				delaunayTri.append((ind[0], ind[1], ind[2]))
			elif len(ind) > 3:
				ind = best_solution(ind,3, 5)
				delaunayTri.append((ind[0], ind[1], ind[2]))
			else:
				logging.error('Insufficient points for making triangle')
		pt = []

	return delaunayTri

'''Reference: https://flothesof.github.io/farthest-neighbors.html'''
def evaluate_solution(solution_set):
	return sum([distance(a, b) for a, b in zip(solution_set[:-1], solution_set[1:])])

def best_solution(points, k, tries):
    solution_sets = [incremental_farthest_search(points, k) for _ in range(tries)]
    sorted_solutions = sorted(solution_sets, key=evaluate_solution, reverse=False)
    return sorted_solutions[0]

def incremental_farthest_search(points, k):
    remaining_points = points[:]
    solution_set = []
    solution_set.append(remaining_points.pop(random.randint(0, len(remaining_points) - 1)))
    for _ in range(k-1):
        distances = [distance(p, solution_set[0]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], distance(p, s))
        solution_set.append(remaining_points.pop(distances.index(max(distances))))
    return solution_set

def distance(A, B):
    return abs(A - B)

def listOfListToTuples(p):
	t_list = []
	for ent in p:
		s = []
		for i in range(0, len(ent)):
			s.append(ent[i])
		t_list.append(tuple(s))

	return t_list