'''
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

import numpy as np
from FFTposetrack import polarTransform
import scipy
import multiprocessing
import sys
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2

def FFTspectrum(image):
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft.copy())
    spectrum = 20.0 * np.abs(fft_shift)
    spectrum -= spectrum.min()
    spectrum /= spectrum.max()

    return spectrum

def threeTofour(mat):
    four = np.eye(4)
    four[:2, :2] = mat[:2, :2]
    four[0, 3] = mat[0, 2]
    four[1, 3] = mat[1, 2]
    return four



def fourTothree(mat):
    three = np.eye(3)
    three[:2, :2] = mat[:2, :2]
    three[0, 2] = mat[0, 3]
    three[1, 2] = mat[1, 3]

    return three

def rot_fft2(image):
    reference_fft = scipy.fft.fft2(image, workers=multiprocessing.cpu_count())
    reference_fft_shift = np.fft.fftshift(reference_fft.copy())
    reference_spectrum = 20.0 * np.abs(reference_fft_shift)
    reference_spectrum -= reference_spectrum.min()
    reference_spectrum /= reference_spectrum.max()

    reference_polar = polarTransform(reference_spectrum)
    reference_polar_theta = np.sum(reference_polar, axis=1)
    reference_theta_fft = np.fft.fft(reference_polar_theta)

    return [reference_polar_theta, reference_theta_fft, reference_spectrum]


def img2local(width, height, x, y):
    return int(x-width/2), int(height/2-y)

def local2img(width, height, x, y):
    return int(x+width/2) , int(height/2-y)

def plotCrosspoint(img, cross_points, width, height):
    for cross_point_dict in cross_points:
        cross_point = cross_point_dict["point"]
        segments = cross_point_dict["segments"]
        ref_A, ref_B, cur_C, cur_D = segments[:]
        cross_x, cross_y = local2img(width, height, cross_point[0], cross_point[1])
        ref_A_x, ref_A_y = local2img(width, height, ref_A[0], ref_A[1])
        ref_B_x, ref_B_y = local2img(width, height, ref_B[0], ref_B[1])
        cur_C_x, cur_C_y = local2img(width, height, cur_C[0], cur_C[1])
        cur_D_x, cur_D_y = local2img(width, height, cur_D[0], cur_D[1])

        cv2.circle(img, (int(cross_x), int(cross_y)), radius=4, color=(255, 0, 0), thickness=-1)
        cv2.line(img, (ref_A_x, ref_A_y), (ref_B_x, ref_B_y), (0, 0, 255), thickness=2)
        cv2.line(img, (cur_C_x, cur_C_y), (cur_D_x, cur_D_y), (0, 0, 255), thickness=2)


    # cv2.imshow("frame", frame)
    cv2.imshow("cross+line", img)
    cv2.waitKey(1)

