

# import the necessary packages

import cv2
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import RPi.GPIO as io

d = time.time()


def euclidean_dist(ptA, ptB):

    # compute and return the euclidean distance between the two
    # points

    return np.linalg.norm(ptA - ptB)


def eye_aspect_ratio(eye):

    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates     

    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates

    C = euclidean_dist(eye[0], eye[3])

    # compute the eye aspect ratio

    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio

    return ear


EYE_AR_THRESH = 0.0
EYE_AR_CONSEC_FRAMES = 12
EYE_AR_CONSEC_FRAMES1 = 1

# initialize the frame counter 

COUNTER = 0
COUNTER1 = 1

# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor

print ('[INFO] loading facial landmark predictor...')


detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat'
                                 )

# grab the indexes of the facial landmarks for the left and
# right eye, respectively

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# start the video stream thread

print ('[INFO] starting video stream ...')


vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)
d3=time.time()
d5=time.time()
d2 = time.time()
cnt2 = 0    #EAR threshold calculation
sum = 0.0
cnt_closed=0;
cnt_open=0;
perclos=0
flag_for_perclos=True #to ensure that perclos calculation doesnt happen in the forst iteration.

# loop over frames from the video stream

while True:

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)

    frame = vs.read()

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # detect faces in the grayscale frame

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the face detections

    for (x, y, w, h) in rects:

        # construct a dlib rectangle object from the Haar cascade
        # bounding box

        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes

        ear = (leftEAR + rightEAR) / 2.0
        if cnt2<=20 and ear > 0:
            d4 = time.time()
            if d4 - d2 >= 0.5:

                d2 = d4
                cnt2 = cnt2 + 1
                sum = sum + ear

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        # print("cnt2 is ",cnt2)

        
        if cnt2==20:
            EYE_AR_THRESH = 5*(sum / cnt2)/6
            print('Threshold is : ',EYE_AR_THRESH)
            cnt2+=1
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0xFF, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0xFF, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter

        
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            cnt_closed+=1


            # if the eyes were closed for a sufficient number of
            # frames, then sound the alarm

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                io.setmode(io.BOARD)
                io.setup(40, io.OUT)
                #GPIO 21
                io.output(40, 1)

                d = time.time()
                cv2.putText(
                    frame,
                    'DROWSINESS ALERT!',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0xFF),
                    2,
                    )
        else:

            COUNTER = 0
            cnt_open+=1
            d1 = time.time()


            if d1 - d > 3:
                io.cleanup()
                d = time.time()

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        d3=time.time()
        if d3-d5>60:
            if flag_for_perclos:
                flag_for_perclos=False
            else:
                perclos=(cnt_closed/(cnt_closed+cnt_open))
                d5=d3
                if perclos>0.05:
                    io.setmode(io.BOARD)
                    io.setup(40, io.OUT)
                    io.output(40, 1)
                    d = time.time()
                    print('perclos =',perclos)
                
                d1 = time.time()
                if d1 - d > 3:
                    io.output(40, 0)
                    io.cleanup()
                    d = time.time()
            perclos=0    
            cnt_closed=0
            cnt_open=0
            counterforframes=0
        cv2.putText(
            frame,
            'EAR: {:.3f}'.format(ear),
            (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0xFF),
            2,
            )

    # show the frame

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    
    if key == ord('q'):
        io.cleanup()
        break


cv2.destroyAllWindows()
vs.stop()