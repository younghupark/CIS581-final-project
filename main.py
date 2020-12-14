from face_detect import *
from feature_extraction import *
from face_swap import *
from cartoonize import *
from optical_flow import *

import cv2
import numpy as np

def main():    
    cap1 = cv2.VideoCapture("./FrankUnderwood.mp4")
    cap2 = cv2.VideoCapture("./MrRobot.mp4")
    frame_cnt = 0
    FRAME_RATE = 5

    # Initialize video writer for tracking video (not working lol)
    trackVideo1 = './results/Output_FrankUnderwood.mp4'
    trackVideo2 = './results/Output_MrRobot.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    size = (1280, 720)
    print("FPS1 and 2", fps1, fps2)
    writer1 = cv2.VideoWriter(trackVideo1, fourcc, fps1, size)
    writer2 = cv2.VideoWriter(trackVideo2, fourcc, fps2, size)

    out = cv2.VideoWriter('output.mp4', fourcc, fps2, size)
    limit = 1000
    points1 = []

    while (cap1.isOpened() and cap2.isOpened()):
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        if not ret1 or not ret2: break
        frame_cnt += 1
        print(frame_cnt)
        img2 = cv2.resize(img2, (1280, 720))

        if ret1 and ret2:
                pos_frame = cap2.get(cv2.CAP_PROP_POS_FRAMES)
                if (pos_frame-1) % FRAME_RATE == 0:
                    # direct face swapping

                    # testing cartoon. comment this out to go back to normal mode
                    # img2 = make_cartoon(img2)
                    # img1 = make_cartoon(img1)

                    # extract landmark points (x, y)
                    landmarks1 = detect_landmarks(img1.copy())[0]
                    landmarks2 = detect_landmarks(img2.copy())[0]

                    # create delauney triangles
                    triangles1 = delauney_triangulation(landmarks1)
                    triangles2 = delauney_triangulation(landmarks2)

                    # transform face frrom source to destinattion
                    img_1_face_to_img_2 = apply_affine_transformation(
                        triangles1, landmarks1, landmarks2, img1, img2)
                    img_2_face_to_img_1 = apply_affine_transformation(
                        triangles2, landmarks2, landmarks1, img2, img1)

                    # apply color correction
                    img_1_face_to_img_2 = correct_colors(img2, img_1_face_to_img_2, landmarks1)
                    img_2_face_to_img_1 = correct_colors(img1, img_2_face_to_img_1, landmarks2)

                    # apply face swap
                    swap_1 = merge_mask_with_image(landmarks2, img_1_face_to_img_2, img2)
                    swap_2 = merge_mask_with_image(landmarks1, img_2_face_to_img_1, img1)

                    cv2.imshow("Blended Swap 1", swap_1)
                    cv2.imshow("Blended Swap 2", swap_2)
                    output = swap_2
                    # # Save video with bbox and all feature points
                    writer1.write(swap_1)
                    writer2.write(swap_2)
                    out.write(swap_2)
                    prev_target_frame = img2
                else:
                    # apply optical flows for the rest of 4 frames
                    output, landmarks2 = doOpticalFlow(output, landmarks2, img2, prev_target_frame, pos_frame)
                    out.write(output)
                    prev_target_frame = img2

        
        # # Save video with bbox and all feature points
        # writer1.write(swap_1)
        # writer2.write(swap_2)

        # Press 'q' on the keyboard to exit
        if cv2.waitKey(30) & 0xff == ord('q'): break

    # Release video reader and video writer
    cv2.destroyAllWindows()
    cap1.release()
    cap2.release()
    writer1.release()
    writer2.release()
    out.release()


if __name__ == "__main__":
    main()
