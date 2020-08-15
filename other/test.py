from math import sqrt, pi

import cv2
import numpy as np
from scipy import stats

def importandgrayscale(path, numFrames, dscale, vidNum):
    # sets up variables to open and save video
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    numFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if path is not 0:
        actualNumFrames = numFrames
    else:
        actualNumFrames = np.inf
    
    # set amount of frames want to look at
    counter = 0
    frame_no = 0
    
    completeVid = np.zeros((int(height/dscale), int(width/dscale), numFrames), dtype=np.uint8)
    
    # main body
    while(cap.isOpened() & (counter < numFrames)):
        ret, frame = cap.read()
        if ((ret == True) & (frame is not None)):
            frame = cv2.resize(frame, (int(width/dscale), int(height/dscale)), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            completeVid[:, :, frame_no] = gray
            frame_no += 1
        else:
            break
        counter += 1

    cap.release()
    print("finshed Black and white conversion")
    return completeVid

def getDirectModeFrame(completeVid):
    # get mode frame of video
    # stats.mode returns mode array as well as mode count, so only first parameter is accessed by modeFrame[0]
    # ModeResult(mode=array([], dtype=float64), count=array([], dtype=float64))
    modeFrame = stats.mode(completeVid, 2)
    modeFrameFinal = modeFrame[0]
    # modeFrameFinal=modeFrameFinal[:,:,0]
    print(modeFrame[0])
    # modeFrameFinal = (modeFrame[0])[:, :, 0]
    print("got direct mode frame")
    return modeFrameFinal

if __name__ == "__main__":
    # cap = cv2.VideoCapture("DJI_0004.MOV")
    completeVid=importandgrayscale(path="DJI_0004.MOV", numFrames=100, dscale=1, vidNum=1)
    modeFrameFinal=getDirectModeFrame(completeVid)
    pass