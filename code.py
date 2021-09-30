#Libraries
import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import torch
#!pip3 install pyttsx3
!pip3 install gtts
import RPi.GPIO as GPIO
import time
GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
set GPIO Pins
GPIO_TRIGGER = 18
GPIO_ECHO = 24
set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
device = torch.device("cuda" if torch.cuda.is_available
() else "cpu")
def distance():
# set Trigger to HIGH
GPIO.output(GPIO_TRIGGER, True)
# set Trigger after 0.01ms to LOW
time.sleep(0.00001)
GPIO.output(GPIO_TRIGGER, False)
StartTime = time.time()
StopTime = time.time()
# save StartTime
while GPIO.input(GPIO_ECHO) == 0:
StartTime = time.time()
A2
# save time of arrival
while GPIO.input(GPIO_ECHO) == 1:
StopTime = time.time()
# time difference between start and arrival
TimeElapsed = StopTime - StartTime
# multiply with the sonic speed (34300 cm/s)
# and divide by 2, because there and back
distance = (TimeElapsed * 34300) / 2
return distance
global dist
if __name__ == '__main__':
try:
while True:
dist = distance()
print ("Measured Distance = %.1f cm" % dist
)
time.sleep(1)
Reset by pressing CTRL + C
except KeyboardInterrupt:
print("Measurement stopped by User")
GPIO.cleanup()
global ob, fdistance
ob = None
fdistance = 0
def draw_labels_and_boxes(img, boxes, confidences, clas
sids, idxs, colors, labels):
# If there are any detections
cdistance = []
j = 0
if len(idxs) > 0:
for i in idxs.flatten():
# Get the bounding box coordinates
x, y = boxes[i][0], boxes[i][1]
w, h = boxes[i][2], boxes[i][3]
global fdistance
fdistance = (917) / (w + h * 360) * 1000 +
3
fdistance = round(fdistance)
# Get the unique color for this class
if fdistance>120:
cdistance.append(fdistance)
else:
cdistance.append(dist)
A3
j = cdistance.index(min(cdistance))
global ob
ob = labels[classids[j]]
color = [int(c) for c in colors[classids[i]
]]
# Draw the bounding box rectangle and label
on the image
cv.rectangle(img, (x, y), (x+w, y+h), color
, 2)
text = "{}: {:.2f} : {:.2f}cm".format(label
s[classids[i]], confidences[i], fdistance)
cv.putText(img, text, (x, y-
5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
return img,ob, fdistance
def generate_boxes_confidences_classids(outs, height, w
idth, tconf):
boxes = []
confidences = []
classids = []
for out in outs:
for detection in out:
# Get the scores, classid, and the confiden
ce of the prediction
scores = detection[5:]
classid = np.argmax(scores)
confidence = scores[classid]
# Consider only the predictions that are ab
ove a certain confidence level
if confidence > tconf:
# TODO Check detection
box = detection[0:4] * np.array([width,
height, width, height])
centerX, centerY, bwidth, bheight = box
.astype('int')
# Using the center x, y coordinates to
derive the top
# and the left corner of the bounding b
ox
x = int(centerX - (bwidth / 2))
y = int(centerY - (bheight / 2))
A4
# Append to list
boxes.append([x, y, int(bwidth), int(bh
eight)])
confidences.append(float(confidence))
classids.append(classid)
return boxes, confidences, classids
def infer_image(net, layer_names, height, width, img, c
olors, labels, FLAGS,
boxes=None, confidences=None, classids=None
, idxs=None, infer=True):
if infer:
# Contructing a blob from the input image
blob = cv.dnn.blobFromImage(img, 1 / 255.0, (41
6, 416),
swapRB=True, crop=False)
# Perform a forward pass of the YOLO object det
ector
net.setInput(blob)
# Getting the outputs from the output layers
outs = net.forward(layer_names)
# Generate the boxes, confidences, and classIDs
boxes, confidences, classids = generate_boxes_c
onfidences_classids(outs, height, width, FLAGS.confiden
ce)
# Apply Non-
Maxima Suppression to suppress overlapping bounding box
es
idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAG
S.confidence, FLAGS.threshold)
if boxes is None or confidences is None or idxs is
None or classids is None:
raise '[ERROR] Required variables are set to No
ne before drawing boxes on images.'
# Draw labels and boxes on the image
img,ob, fdistance = draw_labels_and_boxes(img, boxe
s, confidences, classids, idxs, colors, labels)
A5
#import pyttsx3
#engine = pyttsx3.init()
#engine.setProperty("rate", 200)
#x = 'There is a' + ob + fdistance + 'centimeters a
t your front'
#engine.say(x)
#engine.runAndWait()
from time import sleep
from gtts import gTTS #Import Google Text to Speech
from IPython.display import Audio #Import Audio met
hod from IPython's Display Class
from IPython.core.display import display
from time import sleep
tts = gTTS('There is a' + ob + str(fdistance) + 'ce
ntimeters at your front') #Provide the string to conver
t to speech
tts.save('1.wav') #save the string converted to spe
ech as a .wav file
sound_file = '1.wav'
display(Audio(sound_file, autoplay=True))
sleep(4)
return img, boxes, confidences, classids, idxs, ob,
fdistance
FLAGS = []
if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights',
type=str,
default='/gdrive/My Drive/Demo/yolov3.weights',
help='Path to the file which contains the weights \
for YOLOv3.')
parser.add_argument('-cfg', '--
config', type=str, default='/gdrive/My Drive/Demo/cfg/y
olov3.cfg', help='Path to the configuration file for th
e YOLOv3 model.')
parser.add_argument('-v', '--videopath',
type=str, default= '/gdrive/My Drive/Demo/test.m
p4', help='The path to the video file')
A6
parser.add_argument('-vo', '--video-outputpath',
type=str, default='/gdrive/My Drive/Demo/bagoutp
ut.mp4', help='The path of the output video file')
parser.add_argument('-l', '--
labels', type=str, default='/gdrive/My Drive/Demo/cocolabels',
help='Path to the file having the \ labels in
a new-line seperated way.')
parser.add_argument('-c', '--
confidence', type=float, default=0.5, help='The model w
ill reject boundaries which has a \ probabiity less tha
n the confidence value. \ default: 0.5')
parser.add_argument('-th', '--
threshold', type=float, default=0.3, help='The threshol
d to use when applying the \ Non-Max Suppresion')
FLAGS, unparsed = parser.parse_known_args()
# Get the labels
labels = open(FLAGS.labels).read().strip().split('\n'
)
# Intializing colors to represent each label uniquely
colors = np.random.randint(0, 255, size=(len(labels),
3), dtype='uint8')
# Load the weights and configutation to form the pret
rained YOLOv3 model
net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.w
eights)
# Get the output layer names of the model
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.get
UnconnectedOutLayers()]
# If both image and video files are given then raise
error
if FLAGS.video_path is None:
print ('Path to video not provided')
elif FLAGS.video_path:
# Read the video
vid = cv.VideoCapture(str(FLAGS.video_path))
height, width, writer= None, None, None
while True:
A7
grabbed, frame = vid.read()
if not grabbed:
break
if width is None or height is None:
height, width = frame.shape[:2]
frame, _, _, _, _, _, _ = infer_image(net, layer_
names, height, width, frame, colors, labels, FLAGS)
if writer is None:
fourcc = cv.VideoWriter_fourcc(*'mp4v')
writer = cv.VideoWriter(FLAGS.video_output_path
, fourcc, 30,(frame.shape[1], frame.shape[0]), True)
writer.write(frame)
print ("[INFO] Cleaning up...")
writer.release()
vid.release()
else:
print("[ERROR] Something's not right...")
