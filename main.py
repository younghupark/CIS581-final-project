from face_detect import *
from feature_extraction import *
from face_swap import *
import cv2


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


def draw_image(img, type):
    cv2.imshow(type, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img1 = cv2.imread('./FrankUnderwood.png')
    img2 = cv2.imread('./MrRobot.png')

    landmarks1 = detect_landmarks(img1.copy())[0]
    landmarks2 = detect_landmarks(img2.copy())[0]

    hull1 = cv2.convexHull(landmarks1)
    hull2 = cv2.convexHull(landmarks2)

    triangles1 = delauney_triangulation(landmarks1)
    triangles2 = delauney_triangulation(landmarks2)

    img_1_face_to_img_2 = apply_affine_transformation(
        triangles1, landmarks1, landmarks2, img1, img2)
    img_2_face_to_img_1 = apply_affine_transformation(
        triangles2, landmarks2, landmarks1, img2, img1)

    swap_1 = merge_mask_with_image(landmarks2, img_1_face_to_img_2, img2)
    swap_2 = merge_mask_with_image(landmarks1, img_2_face_to_img_1, img1)

    # draw_convex_hull(hull1, img1.copy())
    # draw_convex_hull(hull2, img2.copy())

    draw_delauney_triangles(triangles1, landmarks1, img1.copy())
    draw_delauney_triangles(triangles2, landmarks2, img2.copy())

    draw_image(swap_1, "Blended Swap")
    draw_image(swap_2, "Blended Swap")


if __name__ == "__main__":
    main()
