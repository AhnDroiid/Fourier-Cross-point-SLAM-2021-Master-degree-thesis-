'''
	This demo file is from an original work by Qidian213/deep sort yolov3 (GPL license 3.0)
	url : https://github.com/Qidian213/deep_sort_yolov3

	Modified it for use in my Master thesis

    Copyright (C) {2020-2021}  {Chanwoo Ahn}
    email : a_chanu0612@snu.ac.kr

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import *
import map

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
	sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import math
import threading
import random
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")

from cv_bridge import CvBridge
import rosbag
import time
from multiprocessing import Process
import csv

avm_topic = "/avm_usb_cam/image_raw"
traj = "avm_vis_1"
bagfile = "/media/aimaster/essd/bagfiles/{}.bag".format(traj)
experiment = "avm_00"
fps = 3

bridge = CvBridge()
bag = rosbag.Bag(bagfile)
warnings.filterwarnings('ignore')

track_list = []
total=0

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
# set_session(tf.Session(config=config))

def pointDetection(full_yolo, Map):
	global track_list
	global total, avm_topic

	# Definition of the parameters
	max_cosine_distance = 0.1
	nn_budget = None
	nms_max_overlap = 1.0

	# deep_sort
	model_filename = 'model_data/mars-small128.pb'
	encoder = gdet.create_box_encoder(model_filename, batch_size=1)

	metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	tracker = Tracker(metric)


	# here define your video file path
	#total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


	currentFrame = 0
	frame_number=1


	for topic, msg, t in bag.read_messages(topics=avm_topic):
		if currentFrame % fps != 0:
			currentFrame += 1
			continue
		if  Map.currentFrame == 0:
			Map.initTime = t.to_sec()
		else : Map.bagTime = t.to_sec() - Map.initTime

		currentFrame += 1
		Map.currentFrame += 1

		data = np.fromstring(msg.data, dtype=np.uint8)
		frame = np.reshape(data, newshape=(320, 480, 3))
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		Map.currentImage = frame.copy()

		image = Image.fromarray(frame)

		boxs = full_yolo.detect_image(image)


		# back side box detection.
		features = encoder(frame, boxs)
		# score to 1.0 here).
		detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

		# Run non-maxima suppression.
		boxes = np.array([d.tlwh for d in detections])
		scores = np.array([d.confidence for d in detections])
		indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
		detections = [detections[i] for i in indices]



		# Call the tracker
		tracker.predict()
		tracker.update(detections)


		for track in tracker.tracks:
			if track.is_confirmed() and track.time_since_update > 1:
				continue


			# bbox consists of [xmin, ymin, xmax, ymax]
			bbox = track.to_tlbr()

			xmin = int(bbox[0])
			xmax = int(bbox[2])

			avg = (xmin + xmax) / 2

			if avg >0 :
				#print(int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3]), "id:", track.track_id) # xmin, ymin , xmax, ymax, feature ID
				# cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
				# cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
				#print("Deep sort id : {}".format(track.track_id))

				Map.buffer.append([int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3]), track.track_id])


		Map.convertToMask(Map.currentImage)
		Map.updatePose()
		Map.draw()
		#Map.drawBoxId()

		frame_number += 1
		# cv2.imshow("mask:", Map.frames[-1])
		# cv2.imshow("frame", frame)
		# cv2.waitKey(1)

	cv2.destroyAllWindows()


if __name__ == '__main__':
	full_yolo = YOLO(model_path='logs/{}/trained_weight_final.h5'.format(experiment), anchor_path='./logs/{}/yolo_anchors.txt'.format(experiment),
	 				 image_size=(320, 480), score=0.12)

	Map = map.Map(img_height=320, img_width=480, traj=traj)

	pointDetection(full_yolo=full_yolo, Map=Map)
