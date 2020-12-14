import cv2
import numpy as np
POLY_FILL_COLOR = (1.0, 1.0, 1.0)


def get_affine_transform(src, src_tri, dst_tri, size):
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_triangular_region(triangle_1, triangle_2, img_1, img_2):
    # Find bounding rectangle for each triangle in the form <x, y, w, h>
    x_1, y_1, w_1, h_1 = cv2.boundingRect(np.float32([triangle_1]))
    x_2, y_2, w_2, h_2 = cv2.boundingRect(np.float32([triangle_2]))

    # Offset points by left top corner of the respective rectangles
    offset_triangle_1 = []
    offset_triangle_2 = []

    # for the <x,y> coordinates of each point the triangle find the offset
    # move this into a separate function if you need to do it a for a lot of triangles
    for coords in triangle_1:
        offset_triangle_1.append(((coords[0] - x_1), (coords[1] - y_1)))
    for coords in triangle_2:
        offset_triangle_2.append(((coords[0] - x_2), (coords[1] - y_2)))

    # get the mask by filling the triangle to mask pixels outside the desired area
    mask = np.zeros((h_2, w_2, 3))
    cv2.fillConvexPoly(mask, np.int32(offset_triangle_2), (1.0, 1.0, 1.0))

    # get only the part of the image we are going to map within the bounding rectangle
    img_1_within_bounds = img_1[y_1:y_1 + h_1, x_1:x_1 + w_1]

    size_bounds_triangle_2 = (w_2, h_2)

    # apply the affine transform on img_1 based on the triangles
    transformed_area = get_affine_transform(img_1_within_bounds, offset_triangle_1, offset_triangle_2,
                                            size_bounds_triangle_2)

    # remove all parts of the transformed image outside the area we care about (triangle mask)
    transformed_triangle = transformed_area * mask

    # slice the current area out of the in the image we are mapping the face to
    img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] = img_2[y_2:y_2 +
                                                h_2, x_2:x_2 + w_2] * (POLY_FILL_COLOR - mask)
    # slice the transformed area back in its place
    img_2[y_2:y_2 + h_2, x_2:x_2 + w_2] = img_2[y_2:y_2 +
                                                h_2, x_2:x_2 + w_2] + transformed_triangle

    return img_2


def apply_affine_transformation(delauney, hull_1, hull_2, img_1, img_2):
    # create a copy of image 2 that we will map the face from image 1 to
    img_2_with_face_1 = np.copy(img_2)

    # morph each triangular region one at a time
    for triangle in delauney:
        triangles_1 = []
        triangles_2 = []

        # get points within img_1 and img_2 corresponding to the triangle points previously found from the face in img_1
        for point in triangle:
            triangles_1.append(hull_1[point])
            triangles_2.append(hull_2[point])

        # once we have found the points in the landmarks corresponding to the triangle morph the triangular region from
        # img_1 to img_2 and return the result that we will modify again with the next triangle
        morph_triangular_region(triangles_1, triangles_2,
                                img_1, img_2_with_face_1)

    return img_2_with_face_1


# Get the area that was transformed in order to seamlessly clone
def calculate_mask(landmarks, img):
    hull_tuples = []

    hull = []
    # this is the area that we will be mapping between faces
    hull_index_to_map = cv2.convexHull(np.array(landmarks), returnPoints=False)

    # find the facial landmark points on both faces that are within the hull of the face we are basing our map off of
    for i in range(0, len(hull_index_to_map)):
        hull.append(landmarks[int(hull_index_to_map[i])])

    for points in hull:
        hull_tuples.append((points[0], points[1]))
    # create a mask that encompasses the whole image
    mask = np.zeros(img.shape, dtype=img.dtype)

    # use the empty mask as the input image and the hull tuples a polygon vertices to fill
    # this fills only the area of the hull
    cv2.fillConvexPoly(mask, np.int32(hull_tuples), (255, 255, 255))

    # cv2.imshow("Image Mask ", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hull_bounding_rectangle = cv2.boundingRect(np.float32([hull]))

    bounding_rectangle_center = (hull_bounding_rectangle[0] + int(hull_bounding_rectangle[2] / 2),
                                 hull_bounding_rectangle[1] + int(hull_bounding_rectangle[3] / 2))

    # return the mask of the face area and the center of the bounding bounding box which contains the face
    return mask, bounding_rectangle_center


# Smooths and blends the mask to look more natural
def merge_mask_with_image(landmarks, img_with_mapped_face, original_img):
    mask, center = calculate_mask(landmarks, original_img)

    # make the mask smaller 
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)

    # poisson blending to smooth out the edges
    return cv2.seamlessClone(np.uint8(img_with_mapped_face), original_img, mask, center, cv2.NORMAL_CLONE)


# Color Correction
def correct_colors(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result