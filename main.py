from face_detect import *
from feature_extraction import *
from face_swap import *
from cartoonize import *
from optical_flow import *
from cartoonize_2 import *

import cv2
import numpy as np

def main():    
    cap1 = cv2.VideoCapture("./videos/FrankUnderwood.mp4")
    cap2 = cv2.VideoCapture("./videos/MrRobot.mp4")
    frame_cnt = 0

    # change the frame rate to 1 if not want to apply Optical Flow
    FRAME_RATE = 1
    # change this to 0 to not cartoonize or 1 to cartoonize
    CARTOONIZE = 1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    size = (1280, 720)

    out = cv2.VideoWriter('./results/cartoonize_of_output.mp4', fourcc, fps2, size)
    out2 = cv2.VideoWriter('./results/cartoonize_of_output2.mp4', fourcc, fps2, size)

    

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
                    if CARTOONIZE:
                        img2 = cartoonize(img2)
                        img1 = cartoonize(img1)
                  

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

                    # cv2.imshow("Blended Swap 1", swap_1)
                    # cv2.imshow("Blended Swap 2", swap_2)

                    output = swap_1
                    output2 = swap_2
                    # # Save video with bbox and all feature points
                    # writer1.write(swap_1)
                    # writer2.write(swap_2)
                    out.write(swap_1) # optical flow applied
                    out2.write(swap_2) # optical flow applied

                    prev_target_frame = img2
                    prev_target_frame_2 = img1
                else:
                    
                    # apply optical flows for the rest of 4 frames
                    output, landmarks2 = get_optical_flow(output, landmarks2, img2, prev_target_frame, pos_frame)
                    output2, landmarks1 = get_optical_flow(output2, landmarks1, img1, prev_target_frame_2, pos_frame)
                    
                    if CARTOONIZE:
                        output2 = cartoonize(output2)
                        output = cartoonize(output)

                    out.write(output)
                    out2.write(output2)
                    prev_target_frame = img2
                    prev_target_frame_2 = img1

        # Press 'q' on the keyboard to exit
        if cv2.waitKey(30) & 0xff == ord('q'): break

    # Release video reader and video writer
    cv2.destroyAllWindows()
    cap1.release()
    cap2.release()
    # writer1.release()
    # writer2.release()
    out.release()
    out2.release()


if __name__ == "__main__":
    main()
