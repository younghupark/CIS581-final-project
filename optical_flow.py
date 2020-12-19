import cv2
import numpy as np
import logging
from skimage.transform import SimilarityTransform, matrix_transform
from feature_extraction import convex_hull

from triangulation import triangulation
from warping import warping

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
					maxLevel = 2,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def transformation(H, x):
	newBbox_temp = np.matmul(H, np.hstack((x, np.ones((x.shape[0], 1))))[:, :, None])
	newBbox_temp = np.squeeze(newBbox_temp)
	newBbox_temp = newBbox_temp[:, 0:2] / newBbox_temp[:, 2][:, None]
	newBbox_temp = np.squeeze(newBbox_temp)
	return newBbox_temp

def get_optical_flow(prevOutput, targetPoints, target_frame, prev_target_frame, frame_no):
	p0 = np.asarray(targetPoints).astype(np.float32)[:, :, None]
	p0 = np.transpose(p0, (0, 2, 1))
	old_gray = cv2.cvtColor(prev_target_frame, cv2.COLOR_BGR2GRAY)
	frame_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]

	newOutput = np.copy(prevOutput)

	transform = SimilarityTransform()
	if transform.estimate(good_old, good_new):
		newOutput = transform_image(good_old, good_new, prevOutput, target_frame, frame_no)
	return newOutput, tuplify(good_new.tolist())

def transform_image(points1, points2, img1, img2, frame_no):
	img1Warped = np.copy(img2)
	hull1, hull2 = convex_hull(points1.tolist(), points2.tolist())
	hull1 = np.array(hull1).astype(np.float32)
	hull2 = np.array(hull2).astype(np.float32)
	if empty_points(hull1, hull2, 2, frame_no): 
		return img2

	hull2 = np.asarray(hull2)
	hull2[:, 0] = np.clip(hull2[:, 0], 0, img2.shape[1] - 1)
	hull2[:, 1] = np.clip(hull2[:, 1], 0, img2.shape[0] - 1)
	hull2 = listOfListToTuples(hull2.astype(np.float32).tolist())

	dt = triangulation(img2, hull2)
	if len(dt) == 0:
		return img2

	warping(dt, hull1, hull2, img1, img1Warped)

	return img1Warped

def empty_points(points1, points2, step, frame_no):
	if len(points1) == 0 or len(points2) == 0:
		return True

	return False

def tuplify(p):
	t_list = []
	for ent in p:
		s = [ent[i] for i in range(len(ent))]
		t_list.append(tuple(s))
	
	return t_list