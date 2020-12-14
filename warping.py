import cv2,numpy as np

def warping(dt, hull1, hull2, source_frame, img1Warped):
	# Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
		# get points for img1, img2 corresponding to the triangles
        t1 = [hull1[dt[i][j]] for j in range(0,3)]
        t2 = [hull2[dt[i][j]] for j in range(0,3)]
        warpTriangle(source_frame, img1Warped, t1, t2)


def warpTriangle(img1, img2, t1, t2):
	# Find bounding rectangle for each triangle
	r1 = cv2.boundingRect(np.float32([t1]))
	r2 = cv2.boundingRect(np.float32([t2]))

	# Offset points by left top corner of the respective rectangles
	t1Rect = [((t1[i][0] - r1[0]), (t1[i][1] - r1[1])) for i in range(0,3)]
	t2Rect = [((t2[i][0] - r2[0]), (t2[i][1] - r2[1])) for i in range(0,3)]
	t2RectInt = [((t2[i][0] - r2[0]), (t2[i][1] - r2[1])) for i in range(0,3)]


	# Get mask by filling triangle
	mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
	cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

	# Apply warpImage to small rectangular patches
	img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

	size = (r2[2], r2[3])
	img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
	img2Rect = img2Rect * mask

	# Copy triangular region of the rectangular patch to the output image
	img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
	(1.0, 1.0, 1.0) - mask)

	img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

def applyAffineTransform(src, srcTri, dstTri, size):
	warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
	dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
						 borderMode=cv2.BORDER_REFLECT_101)

	return dst