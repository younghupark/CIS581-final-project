import cv2

def extractImg(video):
    """
    Extract the first image of a given video and save it as a png
    """
    cap = cv2.VideoCapture(video)
    video_name = video.split('/')[-1].split('.')[0] + '.png'
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imwrite(video_name, frame)
        break

if __name__=="__main__":
    video1 = "./final_demo/FrankUnderwood.mp4"
    video2 = "./final_demo/MrRobot.mp4"
    extractImg(video1)
    extractImg(video2)
