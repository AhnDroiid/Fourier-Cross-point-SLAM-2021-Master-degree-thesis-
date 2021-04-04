
'''
    TODO: Upgrade this program to use particle filter

	This demo file is from an original work by nwang57/FastSLAM
	url : https://github.com/nwang57/FastSLAM

	Modified it for my academic purpose
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
from numpy import sin, cos
import numpy as np

class EKF(object):
    def __init__(self, rad, dx, dy, robot=False, crosspoint_cov=None):
        '''

        Parameters
        ----------
        initialized world coordinate

        rad : orientation of object  (radian, counterclockwise positive)
        dx :  x position of object (pixel format)
        dy :  y position of object (pixel format)
        '''

        self.isRobot = robot
        if robot:
            self.dx, self.dy, self.rad = dx, dy, rad
            self.pose_cov = np.array([[10e-15, 0, 0], [0, 10e-15, 0], [0, 0, 10e-15]]) # 3x3

        else:
            if crosspoint_cov is None:
                print("Error: if tracker is for cross point. It should have initial covariance")
                exit(-1)
            self.dx, self.dy = dx, dy
            self.crosspoint_cov = crosspoint_cov # 2x2

        self.obs_noise = np.array([[3, 0], [0, 3]]) # 2x2

    def robot_computeJacobian(self, EKFcrosspoint):
        '''

        Parameters
        ----------
        predRobotPose
        EKFcrosspoint

        Returns
        -------

        '''

        rad, dx, dy = self.rad, self. dx, self.dy
        world_x, world_y = EKFcrosspoint.dx, EKFcrosspoint.dy

        pred_x, pred_y = cos(rad) * (world_x - dx) + sin(rad) * (world_y-dy), -sin(rad) * (world_x - dx) + cos(rad) * (world_y - dy)
        crossPointJac = np.array([[cos(rad), sin(rad)], [-sin(rad), cos(rad)]], dtype="float64") # dx ,dy // 2x2
        poseJac = np.array([[-sin(rad) * (world_x - dx), -cos(rad), -sin(rad)], [-cos(rad) * (world_x - dx), sin(rad), -cos(rad)]], dtype="float64") # rad, dx, dy// 2x3
        adjCov = crossPointJac @ EKFcrosspoint.crosspoint_cov @ crossPointJac.T + self.obs_noise # 2x2

        return np.array([[pred_x], [pred_y]]), crossPointJac, poseJac, adjCov

    def crosspoint_computeJacobian(self, EKFrobot):
        '''

        Parameters
        ----------
        predRobotPose
        EKFcrosspoint

        Returns
        -------

        '''

        rad, dx, dy = EKFrobot.rad, EKFrobot.dx, EKFrobot.dy

        world_x, world_y = self.dx, self.dy

        pred_x, pred_y = cos(rad) * (world_x - dx) + sin(rad) * (world_y-dy), -sin(rad) * (world_x - dx) + cos(rad) * (world_y - dy)
        crossPointJac = np.array([[cos(rad), sin(rad)], [-sin(rad), cos(rad)]], dtype="float64") # dx ,dy // 2x2
        poseJac = np.array([[-sin(rad) * (world_x - dx), -cos(rad), -sin(rad)], [-cos(rad) * (world_x - dx), sin(rad), -cos(rad)]], dtype="float64") # rad, dx, dy// 2x3
        adjCov = crossPointJac @ self.crosspoint_cov @ crossPointJac.T + self.obs_noise # 2x2

        return np.array([[pred_x], [pred_y]]), crossPointJac, poseJac, adjCov

    def robot_predict(self, predRobotPose, ):
        self.rad, self.dx, self.dy = predRobotPose

        self.pose_cov = np.array([[np.pi / 300, 0, 0], [0, 10e-10, 0], [0, 0, 10e-5]]) # 3x3

    def robot_update(self, EKFcrosspoint, obs):
        '''

        Parameters
        ----------
        predRobotPose : pred robot's world position
        EKFcrosspoint : Crosspoint's EKF tracker object
        obs : actual observation of cross point in image coordinate  2x1
        Returns
        -------

        '''

        pred, crossPointJac, poseJac, adjCov = self.robot_computeJacobian(EKFcrosspoint)
        self.pose_cov = np.linalg.inv( poseJac.T @ np.linalg.inv(adjCov) @  poseJac + np.linalg.inv(self.pose_cov) )

        rad, dx, dy = self.rad, self.dx, self.dy

        # 3x1 + 3x1
        pose_mean = np.array([[rad], [dx], [dy]]) + \
                    self.pose_cov @ poseJac.T @ np.linalg.inv(adjCov) @ ( obs - pred )

        self.rad, self.dx, self.dy = np.random.multivariate_normal(pose_mean[:, 0], self.pose_cov)

    def crosspoint_update(self, EKFrobot, obs):
        '''

        Parameters
        ----------
        EKFrobot
        obs

        Returns
        -------

        '''
        pred, crossPointJac, poseJac, adjCov = self.crosspoint_computeJacobian(EKFrobot)

        KalmanGain = self.crosspoint_cov @ crossPointJac.T @ np.linalg.inv(adjCov)
        updatedVal = np.array([[self.dx], [self.dy]], dtype="float64") + KalmanGain @ (obs - pred)
        self.crosspoint_cov = (np.eye(2) - KalmanGain @ crossPointJac) @ self.crosspoint_cov
        self.dx, self.dy = updatedVal[0, 0], updatedVal[1, 0]

