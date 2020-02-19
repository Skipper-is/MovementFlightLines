import cv2
import numpy as np
import time


def ChangeColour(img):
    h,w = np.shape(img)
    county = 0
    for y in img:
        countx = 0
        for px in y:
            if px >0:
                px = px-1
            img[county,countx] = px
            countx = countx+1
        county = county +1
    return img

def colourMap(px):
    if px > 0:
        return px-1
    else:
         return 0


white = (255,255,255)
grey = (128,128,128)
black = (0,0,0)
cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorKNN()
cap.set(28,0)
ret,frame = cap.read()
prevMask = None
#cap.set(15,-20)
time.sleep(2)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
changeFunc = np.vectorize(colourMap,otypes=[np.uint8])
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('E:\Documents\GitHub\PythonScraps\ObjectRecognition\output.mp4',cv2.VideoWriter_fourcc(*"mp4v"), 10, (frame_width,frame_height))


while(1):
    ret, frame = cap.read()
    fgmask = backSub.apply(frame)
    if prevMask is not None:
        dst = cv2.bitwise_or(prevMask,fgmask)
        invertedDST = (255-dst)
        invertedFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        invertedDST = cv2.addWeighted(invertedFrame,0.8,invertedDST,0.2,0)
        cv2.imshow("Change",invertedDST)
        writerFrame = cv2.cvtColor(invertedDST, cv2.COLOR_GRAY2BGR)
        out.write(writerFrame)
        #prevMask = ChangeColour(dst)
        prevMask = dst
        
        #prevMask = changeFunc(dst)
        
    elif prevMask is None:
        prevMask = fgmask
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("Original",frame)

    cv2.imshow("Edges",fgmask)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k == 32:
        prevMask = None
        continue



cap.release()
out.release()
cv2.destroyAllWindows()
