'''
    Map structure definition

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
import sys

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
from numpy import cos as cos
from numpy import sin as sin
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time
from matplotlib.patches import Rectangle
from FFTposetrack import posetrack, getGlobalPose, getUncertainty_2, getInterRotationUncertainty
from scipy.ndimage import rotate
from sklearn.neighbors import NearestNeighbors
from utils import *
from collections import OrderedDict
from ExtendedKalmanTracker import *
import random
import pygame
from pygame.locals import *
from pylsd import lsd

def returnCoord(image):
    row, col = np.where(image != 0)
    x, y = np.expand_dims(col - image.shape[1] / 2, -1), np.expand_dims(image.shape[0] / 2 - row, -1)
    z = np.expand_dims(np.zeros_like(row), -1)
    # Calculate x, y coordinate w.r.t image center point

    coord = np.concatenate([x, y, z], axis=-1).astype(np.float64)

    return coord


def high_Intensity(image):
    image_roi = image[78:245,103:373 ,:]
    roi_copy = image_roi.copy()

    # get hls color space
    hls =  cv2.cvtColor(image_roi, cv2.COLOR_RGB2HLS)
    L_channel = hls[:, :, 1]
    mask = cv2.inRange(L_channel, 180, 255)
    image_roi = cv2.bitwise_and(image_roi, image_roi, mask=mask)

    image_roi_gray = cv2.cvtColor(image_roi.copy(), cv2.COLOR_RGB2GRAY)
    image_roi_gray[image_roi_gray>170] = 255
    image_roi_gray[image_roi_gray<170] = 0
    image_roi_gray[71:115, 155:210] = 0
    image_roi_gray = image_roi_gray[:, 55:]
    image_roi_gray = image_roi_gray[:, :156]

    roi_copy = roi_copy[:, 55:, :]
    roi_copy = roi_copy[:, :156, :]

    #image_roi_gray = cv2.Canny(image_roi_gray, 10, 255)

    return roi_copy, image_roi_gray

def getCoord(image):
    row, col = np.where(image==255)
    return np.array([[i, j] for i, j in zip(row, col)])

def backToImage(coord, w, h):
    image = np.zeros((h, w))
    #print(image.shape)
    for i in range(coord.shape[0]):
        y, x = int(coord[i][0]), int(coord[i][1])
        if y < h and y >= 0 and x < w and x >=0:
            image[y, x] = 255
    return image


class Map():
    def __init__(self, img_width, img_height, traj):
        self.traj = traj
        self.poses = [[0, 0, 0]] # radian, dx, dy
        self.robotTracker = EKF(0, 0, 0, robot=True)
        self.crossPointpose = OrderedDict([]) # "id" : [x,y, frame_id] (center coordinate)
        self.buffer = []
        self.currentFrame = 0
        self.img_width = img_width
        self.img_height = img_height
        self.bagTime = 0
        self.initTime = 0
        self.currentImage = None
        self.frames = []
        self.fft2frames = []
        self.translationKeyFrames = [0]
        self.rotationKeyFrames = [0]
        self.Globalyaws = [0]
        self.Globaltrans = [[np.array([0, 0]).T]]
        self.stdUncertainty = None

        self.Xscale = 2.01 / 52
        self.Yscale = 2.1 / 50

        self.deg2rad = np.pi / 180
        self.rad2deg = 180 / np.pi

        self.iouThreshold = 0.15
        self.matchedBuffer = []
        self.unmatchedBuffer = []
        self.nn = NearestNeighbors(n_neighbors=1, metric="euclidean")

        self.FPS = 30
        self.WindowWidth = 1000
        self.WindowHeight = 1000
        self.pygame = pygame
        self.pygame.init()
        self.window = self.pygame.display.set_mode((self.WindowWidth, self.WindowHeight), 0, 32)
        self.pygame.display.set_caption("FT Crosspoint SLAM")
        self.fpsClock = self.pygame.time.Clock()
        self.mainClock = self.pygame.time.Clock()
        self.font = self.pygame.font.Font(None, 20)

        self.carWidth, self.carHeight = 100, 40
        self.image = self.pygame.transform.scale(self.pygame.image.load("car.jpeg"), (self.carWidth, self.carHeight))


        self.COLOR = { "white": (255, 255, 255),
          "black": (0, 0, 0),
          "green": (0, 255, 0),
          "blue": (0, 0, 255),
          "red": (255, 0, 0),
          "purple": (128, 0, 128)
        }
        self.drawingBuffer = []
        #self.file = open("/media/aimaster/essd/estimates/{}/vert.txt".format(self.traj), "w")

    def draw(self):
        for event in self.pygame.event.get():
            if event.type == QUIT:
                self.pygame.quit()
                sys.exit()

        self.window.fill(self.COLOR["white"])

        rad, dx, dy = self.poses[-1]
        py_w, py_h = self.convertToPygameCoord(dx, dy)
        image = pygame.transform.rotate(self.image, rad * self.rad2deg)

        self.window.blit(image, (py_w- self.carWidth / 2, py_h - self.carHeight / 2))
        frame = self.currentImage.copy()
        for id, xmin, ymin, xmax, ymax in self.drawingBuffer:
            world_x, world_y = self.crossPointpose[id][:2]
            py_w, py_h = self.convertToPygameCoord(world_x, world_y)
            self.window.blit(self.font.render(str(id), True, self.COLOR["black"]), (py_w-10, py_h-10))
            self.pygame.draw.circle(self.window, self.COLOR["blue"], (py_w, py_h), 5)
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), 2)
            cv2.putText(frame, str(id), (int(xmin), int(ymin)), 0, 5e-3 * 200, (0, 255, 0), 2)
        self.drawingBuffer = []
        self.fpsClock.tick(self.FPS)
        self.pygame.display.update()
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    def convertToMask(self, image):
        img = image[78:245,103:373 ,:].copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        segments = lsd(img_gray, scale=0.5)

        for i in range(segments.shape[0]):
            pt1 = (int(segments[i, 0]), int(segments[i, 1]))
            pt2 = (int(segments[i, 2]), int(segments[i, 3]))
            width = segments[i, 4]
            cv2.line(img, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))

        img_roi, mask = high_Intensity(image)
        self.frames.append(mask)

        #cv2.imshow("lsd", img)
        #cv2.imshow("mask", mask)
        #cv2.waitKey(1)

        self.fft2frames.append(rot_fft2(mask))

    def convertTocenter(self, image_h, image_w):
        # In avm center coordinate right is x , up is y direction
        # image_h : height direction position, image_w : width direction position
        return image_w - self.img_width / 2 , self.img_height / 2 - image_h
    def convertToPygameCoord(self, world_x, world_y):
        return  (int(world_x + self.WindowWidth / 2),  int(self.WindowHeight / 2 - world_y))

    def getTransformation(self, index):
        rad, dx, dy = self.poses[index]
        return np.array([[cos(rad), -sin(rad), 0, dx],
                         [sin(rad), cos(rad), 0, dy],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


    def registerMatchedCrosspoint(self, crossPoint):
        # register single cross point in self.crossPointpose using current vehicle pose(self.poses[-1])
        xmin, ymin , xmax, ymax, sortId = crossPoint
        if sortId in list(self.crossPointpose.keys()) :
            image_h, image_w = (ymin + ymax) / 2, (xmin + xmax) / 2
            center_x, center_y = self.convertTocenter(image_h, image_w)
            self.matchedBuffer.append([sortId, center_x, center_y])
            self.drawingBuffer.append([sortId, xmin, ymin, xmax, ymax])
            return
        self.unmatchedBuffer.append(crossPoint)

    def registerUnmatchedCrosspoint(self):
        if len(self.unmatchedBuffer) == 0 : return

        for crossPoint in self.unmatchedBuffer:
            xmin, ymin , xmax, ymax, sortId = crossPoint
            image_h, image_w = (ymin + ymax) / 2, (xmin + xmax) / 2
            center_x, center_y = self.convertTocenter(image_h, image_w)
            center_xmin, center_ymin = self.convertTocenter(xmin, ymin)
            center_xmax, center_ymax = self.convertTocenter(xmax, ymax)

            world_x, world_y = (self.getTransformation(-1) @ np.array([[center_x], [center_y], [0.0], [1.0]]))[:2, 0]
            world_xmin, world_ymin = (self.getTransformation(-1) @ np.array([[center_xmin], [center_ymin], [0.0], [1.0]]))[:2, 0]
            world_xmax, world_ymax = (self.getTransformation(-1) @ np.array([[center_xmax], [center_ymax], [0.0], [1.0]]))[:2, 0]
            if len(list(self.crossPointpose.keys())) == 0:
                cpTracker = EKF(None, world_x, world_y, robot=False, crosspoint_cov=self.robotTracker.pose_cov[1:, 1:])
                self.crossPointpose[sortId] = [world_x, world_y, [world_xmin, world_ymin, world_xmax, world_ymax], self.currentFrame, cpTracker]
                self.drawingBuffer.append([sortId, xmin, ymin, xmax, ymax])
                continue

            self.nn.fit(np.array(list(self.crossPointpose.values()))[:, :2].reshape(-1, 2))
            distance, nnCrosspointId = self.nn.kneighbors(np.array([[world_x, world_y]]), return_distance=True)
            # if matched id in self.matchedBuffer >=2 localization data is considerably accurate buf <=1, localization has some uncertainty.

            # high number of identity number is due to increasing number of id number in Deep sort module
            #print("distance : {}, unmatch Id: {}, nn index: {}".format(distance[0][0], sortId, list(self.crossPointpose.keys())[nnCrosspointId[0][0]]))

            if distance[0][0] <= 20:
                matchId = list(self.crossPointpose.keys())[nnCrosspointId[0][0]]
                self.matchedBuffer.append([matchId, center_x, center_y])
                self.drawingBuffer.append([matchId, xmin, ymin, xmax, ymax])
            else:
                cpTracker = EKF(None, world_x, world_y, crosspoint_cov=self.robotTracker.pose_cov[1:, 1:])

                self.crossPointpose[sortId] = [world_x, world_y, [world_xmin, world_ymin, world_xmax, world_ymax], self.currentFrame, cpTracker]
                self.drawingBuffer.append([sortId, xmin, ymin, xmax, ymax])

        self.unmatchedBuffer = []



    def fourierVO(self):
        # Track pose w.r.t last keyframe
        rot_keyRad = self.poses[self.rotationKeyFrames[-1]][0]
        rot_keyAngle = rot_keyRad * self.rad2deg

        trans_keyDx, trans_keyDy = self.poses[self.translationKeyFrames[-1]][1:]
        trans_keyRad = self.poses[self.translationKeyFrames[-1]][0]
        trans_keyAngle = trans_keyRad * self.rad2deg


        #degree
        rot_estim = posetrack(self.frames[self.rotationKeyFrames[-1]], self.frames[-1],
                              self.fft2frames[self.rotationKeyFrames[-1]], self.fft2frames[-1], 0, mode="rotation")
        rel_rot_angle, rot_uncertainty, rot_std_uncertainty, rotCov = rot_estim[0], rot_estim[-3], rot_estim[-2], rot_estim[-1]

        current_global_angle = rot_keyAngle + rel_rot_angle * -1

        # estimate rotation error w.r.t first frame


        adjustment = \
            posetrack(self.frames[0], rotate(self.frames[-1], angle=current_global_angle, reshape=False),
                      self.fft2frames[0], rot_fft2(rotate(self.frames[-1], angle=current_global_angle, reshape=False)), 0, mode="rotation")

        adjustment, adj_uncertainty, adj_std_uncertainty, adjCov = adjustment[0], adjustment[-3], adjustment[-2], adjustment[-1]


        #cv2.imshow("unregistrated frame:", self.fft2frames[0][2] + rotate(self.fft2frames[-1][2], angle=current_global_angle, reshape=False))

        #if adj_uncertainty / adj_std_uncertainty < 1.1:

        # adjust current angle. Recover error..
        #print("adjustment : {}, current global angle: {}".format(adjustment, current_global_angle))
        current_global_angle += -1 * adjustment

        # cv2.imshow("reference frame:", self.fft2frames[0][2])
        # cv2.imshow("target frame:", self.fft2frames[-1][2])
        # cv2.imshow("registrated frame:", self.fft2frames[0][2] + rotate(self.fft2frames[-1][2], angle=current_global_angle, reshape=False))
        # cv2.waitKey(1)

        trans_relativeAngle = trans_keyAngle - current_global_angle

        dx, dy, xCov, yCov = posetrack(self.frames[self.translationKeyFrames[-1]], self.frames[-1], None, None, trans_relativeAngle,
                           mode="translation")

        keyTocurrentUnc = getUncertainty_2(self.frames[self.translationKeyFrames[-1]],
                                           rotate(self.frames[-1], angle=trans_relativeAngle, reshape=False))

        new_rad = self.deg2rad * current_global_angle
        trans_relativeRad = trans_relativeAngle * self.deg2rad

        current_global_trans = np.array([[np.cos(trans_keyRad), -1 * np.sin(trans_keyRad),  trans_keyDx],          \
                                        [np.sin(trans_keyRad),   np.cos(trans_keyRad),      trans_keyDy],          \
                                        [0,                                   0,                      1]])          \
                                                                                                                    \
                                                             @                                                      \
                                                                                                                    \
                               np.array([[np.cos(trans_relativeRad), -1 * np.sin(trans_relativeRad),  dx],          \
                                        [np.sin(trans_relativeRad),   np.cos(trans_relativeRad),      dy],          \
                                        [0,                                   0,                      1]])

        new_dx, new_dy = current_global_trans[0, 2], current_global_trans[1, 2]

        # insert pose queue [global_yaw, global_x, global_y, local_yaw, local_x, local_y, frame_id, keyframe_id, poseUncertainty]


        #print("FT global angle:", current_global_angle, new_dx * self.scale, new_dy * self.scale )

        # update pose
        self.poses.append([new_rad, new_dx, new_dy])

        curRad, curDx, curDy = self.poses[-1]
        prevRad, prevDx, prevDy = self.poses[-2]

        relRad, relDx, relDy = curRad - prevRad, curDx - prevDx, curDy - prevDy

        #rotinterFrameUncertainty = getInterRotationUncertainty(reference_fft2=self.fft2frames[-2], current_fft2=self.fft2frames[-1])

        rot_poseUncertainty = rot_uncertainty / rot_std_uncertainty
        trans_poseUncertainty = keyTocurrentUnc / self.stdUncertainty

        #print(rot_std_uncertainty, rot_uncertainty)

        #print("Rotation Uncertainty : {}, \n Translation Uncertainty: {}".format(rot_poseUncertainty, trans_poseUncertainty))
        #print("rot Cov: {}, x Cov: {}, y Cov:{}".format(rotCov, xCov, yCov))

        #self.rotationKeyFrames.append(self.currentFrame - 1)
        #self.translationKeyFrames.append(self.currentFrame - 1)



        if rot_poseUncertainty > 2:
            self.rotationKeyFrames.append(self.currentFrame - 1)

        if trans_poseUncertainty > 2 and abs(dx / self.Xscale) > 1:
            self.translationKeyFrames.append(self.currentFrame - 1)


    def twoPointLocalization(self):
        if len(self.matchedBuffer) < 2:
            self.registerUnmatchedCrosspoint()
            if len(self.matchedBuffer) < 2: return

        worldPoints = np.zeros(shape=(2, len(self.matchedBuffer)))
        localPoints = np.zeros(shape=(2, len(self.matchedBuffer)))

        for idx, item in enumerate(self.matchedBuffer):
            matchId, center_x, center_y = item
            world_x, world_y = self.crossPointpose[matchId][:2]
            worldPoints[:, idx] = np.array([world_x, world_y], dtype="float64")
            localPoints[:, idx] = np.array([center_x, center_y], dtype="float64")
        length = len(self.matchedBuffer)
        #print("length: {}".format(length))

        world_xCent, world_yCent = np.sum(worldPoints[0, :]) / length, np.sum(worldPoints[1, :]) / length
        local_xCent, local_yCent = np.sum(localPoints[0, :]) / length, np.sum(localPoints[1, :]) / length

        worldCentroid = np.repeat(np.array([[world_xCent], [world_yCent]]), repeats=length, axis=1)
        localCentroid = np.repeat(np.array([[local_xCent], [local_yCent]]), repeats=length, axis=1)

        worldPoints -= worldCentroid
        localPoints -= localCentroid

        H = localPoints @ worldPoints.T

        u, s, v_t = np.linalg.svd(H)

        r = v_t.T @ u.T # 2x2

        rad = self.poses[-1][0]
        r = np.array([[cos(rad), -sin(rad)],
                      [sin(rad), cos(rad)]], dtype="float64")

        t = worldCentroid - r @ localCentroid # 2 x length

        #rad = math.atan2(r[1, 0], r[0, 0])
        #rad = math.acos(r[0, 0])
        dx, dy = t[:, 0]
        self.poses[-1] = [rad, dx, dy]
        self.registerUnmatchedCrosspoint()
        #self.matchedBuffer = []

        #print("degree:{}, dx:{}, dy:{}".format(rad * self.rad2deg, dx*self.scale, dy*self.scale))


    def updatePose(self):
        # Calculate current pose using detected(tracked) cross points
        # cross points data structure : bbox (xmin, ymin , xmax, ymax,, id)

        if self.currentFrame == 1:
            print("World coordinate is set. Just start register detected cross points")
            for i in range(len(self.buffer)):self.registerMatchedCrosspoint(self.buffer[i])
            self.buffer = []
            self.registerUnmatchedCrosspoint()

            self.stdUncertainty = getUncertainty_2(self.frames[0], self.frames[0])
            return

        for i in range(len(self.buffer)):
            self.registerMatchedCrosspoint(self.buffer[i])
        self.buffer = []

        self.fourierVO()
        self.twoPointLocalization()

        self.registerUnmatchedCrosspoint()



        self.matchedBuffer = []

        curRad, curDx, curDy = self.poses[-1]

        print(self.bagTime, curRad * self.rad2deg, curDx * self.Xscale, curDy * self.Yscale)
        #print(len(list(self.crossPointpose.keys())))
        #self.file.write("{} {} {} {} {} {} {} {}\n".format(self.bagTime, curDx * self.Xscale, curDy * self.Yscale, 0, 0, 0, 0, 1))
        #self.file.flush()


        return




