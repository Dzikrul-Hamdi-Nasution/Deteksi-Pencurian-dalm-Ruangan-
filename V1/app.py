# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
#import telepot
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
token = '5751445359:AAFzaIht-oboCxA8RAuJPct4kBtDqR7a9vM'
#id_penerima = 1441844129
id_penerima = 1441844129

lokasi = 'History'
if not os.path.exists(lokasi):
	print('Lokasi Cache: ', lokasi)
	os.makedirs(lokasi)


#bot = telepot.Bot(token)

ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.2,
    help="minimum probability to filter weak detections",
)
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = [
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "Keyboard",
    "p7",
    "p8",
    "p9",
    "p10",
    "p11",
    "p12",
    "p13",
    "p14",
    "p15",
    "person",
    "p16",
    "Monitor",
    "p18",
    "p19",
    "Monitor",
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0) 
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1023)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #take screenshot
            #cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            # draw the prediction on the frame
            
            

            nama=CLASSES[idx]
            nilai=(confidence * 100)-2

            
            if nama == "person" :
                if nilai >85 :
                    label = "{}: {:.2f}%".format(CLASSES[idx], (confidence * 100)-2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
                    )
                    print("Pencurian")
                    # cv2.imwrite(lokasi+'/{}.jpg'.format(nama),frame)
                    # bot.sendMessage(id_penerima,"Terjadi pencurian "+nama)
                    # bot.sendPhoto(id_penerima, photo=open('History/'+nama+'.jpg','rb') )

            if nama == "Monitor" :
                if nilai >30 :
                    label = "{}: {:.2f}%".format(CLASSES[idx], (confidence * 100)-2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
                    )
                    print("Monitor")
                    # cv2.imwrite(lokasi+'/{}.jpg'.format(nama),frame)
                    # bot.sendMessage(id_penerima,"Terjadi pencurian "+nama)
                    # bot.sendPhoto(id_penerima, photo=open('History/'+nama+'.jpg','rb') )
            
            if nama == "Keyboard" :
                if nilai >60 :
                    label = "{}: {:.2f}%".format(CLASSES[idx], (confidence * 100)-2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
                    )
                    print("Keyboard")
                    # cv2.imwrite(lokasi+'/{}.jpg'.format(nama),frame)
                    # bot.sendMessage(id_penerima,"Terjadi pencurian "+nama)
                    # bot.sendPhoto(id_penerima, photo=open('History/'+nama+'.jpg','rb') )
            
            
              
           

            #nilai diatas 70% = trigger

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()


# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
