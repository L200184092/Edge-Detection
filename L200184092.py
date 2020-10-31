import cv2
import numpy
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip

filename = "L200184092.mp4"
codec = cv2.VideoWriter_fourcc("M", "P", "4", "V")
framerate = 30
resolution = (640, 480)

VideoFileOutput = cv2.VideoWriter(filename, codec, framerate, resolution)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    canny = cv2.Canny(blur, 400, 50)
    ret,thresh = cv2.threshold(canny,75,100,0)
    contours, hie = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    color_frame = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    color = cv2.drawContours(color_frame, contours, -1, (255, 255, 255), 1)

    canny1 = cv2.Canny(blur, 300, 50)
    ret,thresh1 = cv2.threshold(canny1,75,100,0)
    contours1, hie = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    color_frame1 = cv2.cvtColor(canny1, cv2.COLOR_GRAY2RGB)
    color1 = cv2.drawContours(color_frame1, contours1, -1, (0, 255, 0), 1)

    canny2 = cv2.Canny(blur, 200, 50)
    ret,thresh2 = cv2.threshold(canny2,75,100,0)
    contours2, hie = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    color_frame2 = cv2.cvtColor(canny2, cv2.COLOR_GRAY2RGB)
    color2 = cv2.drawContours(color_frame2, contours2, -1, (0, 0, 255), 1)

    canny3 = cv2.Canny(blur, 100, 50)
    ret,thresh3 = cv2.threshold(canny3,75,100,0)
    contours3, hie = cv2.findContours(thresh3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    color_frame3 = cv2.cvtColor(canny3, cv2.COLOR_GRAY2RGB)
    color3 = cv2.drawContours(color_frame3, contours3, -1, (255, 0, 0), 1)

    dst = cv2.addWeighted(color_frame,1,color_frame1,1,5)
    dst1 = cv2.addWeighted(color_frame2,1,color_frame3,1,5)
    dst2 = cv2.addWeighted(dst,1,dst1,1,5)
    
    cv2.imshow('dst2',dst2)

    VideoFileOutput.write(dst2)
    
    
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
VideoFileOutput.release()
cap.release()
