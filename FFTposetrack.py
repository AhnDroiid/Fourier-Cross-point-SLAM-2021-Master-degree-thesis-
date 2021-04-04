'''
    Image Registration using Fourier Transform

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
import cv2
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.ndimage import map_coordinates
import time
import scipy
import multiprocessing

def getGlobalPose(prev_GlobalPose, relativePose):
    """
    :param prev_GlobalPose: Global transformation matrix until last keyframe
    :param relativePose: current frame's relative transformation matrix w.r.t last keyframe
    :return: current frame's Global transformation matrix (3x3)
    """
    return np.matmul(prev_GlobalPose, relativePose)

def polarToCartesian(r, theta, width, height):

    x =  (width/2) + r * np.cos(theta)
    y =  (height/2) - r * np.sin(theta)

    return x, y

def polarTransform(image):
    """Return polar transformed image"""
    height, width = image.shape

    rmax = np.sqrt((height/2)**2 + (width/2)**2)
    phimax = np.pi

    rs, thetas = np.meshgrid(np.linspace(0, rmax, width), np.linspace(0, phimax, height))

    xs, ys = polarToCartesian(rs, thetas, width, height)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    coords = np.vstack((ys, xs))

    polarMap = map_coordinates(image, coords, order=3)
    polarMap = polarMap.reshape(height, width)

    return np.flipud(polarMap)


def getSpectAngle(spectrum):
    first_lines = None
    second_lines = None
    first_angle = None
    second_angle = None
    threshold = 10
    eps = 1e-15
    spectrum = spectrum.copy().astype(np.uint8)
    while threshold > 0:
        if first_lines is not None and second_lines is not None: break
        if first_lines is None:
            first_lines = cv2.HoughLines(spectrum, 1, np.pi / 180, threshold, min_theta=0, max_theta=np.pi/2)
        if second_lines is None:
            second_lines = cv2.HoughLines(spectrum, 1, np.pi / 180, threshold, min_theta=np.pi/2+eps, max_theta=np.pi)
        threshold -= 1

    if first_lines is not None: first_angle = first_lines[:, 0, 1][0] * 180 / np.pi
    if second_lines is not None: second_angle = second_lines[:, 0, 1][0] * 180 / np.pi
    return first_angle, second_angle

def getUncertainty_2(frame_1, frame_2):

    reference_fft = scipy.fft.fft2(frame_1, workers=multiprocessing.cpu_count())
    current_fft = scipy.fft.fft2(frame_2, workers=multiprocessing.cpu_count())

    eps = abs(current_fft).max() * 1e-15
    phase_shift = (reference_fft * current_fft.conjugate()) / (abs(reference_fft) * abs(current_fft) + eps)

    impulse_map = abs(np.fft.ifft2(phase_shift))
    impulse_map -= impulse_map.min()
    impulse_map /= impulse_map.max()

    _impulse_map = np.reshape(impulse_map, newshape=(impulse_map.shape[0] * impulse_map.shape[1]))

    return np.var(_impulse_map)

def getInterRotationUncertainty(reference_fft2, current_fft2):
    reference_polar_theta, reference_theta_fft, _ = reference_fft2[:]
    current_polar_theta, current_theta_fft, _ = current_fft2[:]

    eps = abs(current_theta_fft).max() * 1e-15

    phase_shift = (reference_theta_fft * current_theta_fft.conjugate()) / (abs(reference_theta_fft) * abs(current_theta_fft) + eps)
    impulse_map = abs(np.fft.ifft(phase_shift))
    impulse_map -= impulse_map.min()
    impulse_map /= impulse_map.max()

    std_shift = (reference_theta_fft * reference_theta_fft.conjugate()) / (
                abs(reference_theta_fft) * abs(reference_theta_fft))
    std_impulse_map = abs(np.fft.ifft(std_shift))
    std_impulse_map -= std_impulse_map.min()
    std_impulse_map /= std_impulse_map.max()


    return np.var(impulse_map)/np.var(std_impulse_map)

def optimize(src, target, initial_guess):

    guess = int(initial_guess)

    count = np.zeros(shape=(src.shape[0]*2))
    error_arr = np.ones(shape=(src.shape[0]*2)) * 1e100

    if guess < 0: count[abs(guess) + src.shape[0]] += 1
    else: count[guess] = 1

    src = src / src.max()
    target = target / target.max()

    #print("Optimize! ")
    for i in range(30):
        #print("guess:", guess)
        if abs(guess) > src.shape[0]/2: break
        if guess > 0:
            r = src[guess:] - target[:-1*guess]
            jacobian = np.gradient(target[:-1*guess], edge_order=2)
            jacobian = np.expand_dims(jacobian, axis=-1)
        elif guess == 0:
            r = src - target
            jacobian = np.gradient(target, edge_order=2)
            jacobian = np.expand_dims(jacobian, axis=-1)
        else:
            r = src[:guess] - target[-1*guess:]
            jacobian = np.gradient(target[-1*guess:], edge_order=2)
            jacobian = np.expand_dims(jacobian, axis=-1)

        error = (np.sum(np.square(r)))
        # for the situation when similar guess value is repeatedly calculated
        if guess < 0:
            count[abs(guess) + src.shape[0]] += 1
            error_arr[abs(guess) + src.shape[0]] = error
        else:
            count[guess] += 1
            error_arr[guess] = error


        new_guess = guess - np.linalg.inv(jacobian.T@jacobian) @ jacobian.T @ (r)


        guess_diff = new_guess - guess

        if guess_diff < 0:  guess -= 1
        elif guess_diff > 0:  guess += 1


    smalled_error_idx = int(np.argmin(error_arr))

    if smalled_error_idx < src.shape[0]:
        guess = smalled_error_idx
    else:
        guess = -1 * (smalled_error_idx - src.shape[0])

    if error_arr[smalled_error_idx] < 5:
        if smalled_error_idx < src.shape[0]:
            return smalled_error_idx, error_arr[smalled_error_idx]
        else:

            return -1 * (smalled_error_idx - src.shape[0]), error_arr[smalled_error_idx]
    else:
        error_plot = {}
        for i in range(guess - 10, guess + 10):
            if abs(i) > src.shape[0] / 2: continue
            if i > 0:
                rr = src[i:] - target[:-1 * i]
            elif i == 0:
                rr = src - target
            else:
                rr = src[:i] - target[-1 * i:]

            error_i = np.sum(np.square(rr))

            error_plot[i] = error_i

        initial_guess_error_idx = int(np.argmin(np.array(list(error_plot.values()))))

        return list(error_plot.keys())[initial_guess_error_idx], list(error_plot.values())[initial_guess_error_idx]






def posetrack(referenced_image, current_image, reference_fft2, current_fft2, relativeAngle, mode):
    """
    :param referenced_image: referenced image
    :param current_image: current image
    :return: (rotation alpha(degree), translation 2D vector), uncertainty
    """
    if mode == "rotation":
        # reference_fft = scipy.fft.fft2(referenced_image, workers=multiprocessing.cpu_count())
        # reference_fft_shift = np.fft.fftshift(reference_fft.copy())
        # reference_spectrum = 20.0 * np.abs(reference_fft_shift)
        # reference_spectrum -= reference_spectrum.min()
        # reference_spectrum /= reference_spectrum.max()
        #
        #
        # current_fft = scipy.fft.fft2(current_image, workers=multiprocessing.cpu_count())
        # current_fft_shift = np.fft.fftshift(current_fft.copy())
        # current_spectrum = 20.0 * np.abs(current_fft_shift)
        # current_spectrum -= current_spectrum.min()
        # current_spectrum /= current_spectrum.max()
        #
        # # Estimate rotation
        # reference_polar = polarTransform(reference_spectrum)
        # current_polar = polarTransform(current_spectrum)
        #
        # reference_polar_theta = np.sum(reference_polar, axis=1)
        # current_polar_theta = np.sum(current_polar, axis=1)
        #
        # reference_theta_fft = np.fft.fft(reference_polar_theta)
        # current_theta_fft = np.fft.fft(current_polar_theta)
        reference_polar_theta, reference_theta_fft, _ = reference_fft2[:]
        current_polar_theta, current_theta_fft, _ = current_fft2[:]

        eps = abs(current_theta_fft).max() * 1e-15

        phase_shift = (reference_theta_fft * current_theta_fft.conjugate()) / (abs(reference_theta_fft) * abs(current_theta_fft) + eps)
        angle_shift = abs(np.fft.ifft(phase_shift))
        angle_shift = np.fft.fftshift(angle_shift)

        Flag = False
        angle_shift_copy = angle_shift.copy()
        alpha_shift = None

        while not Flag:
            alpha = np.argmax(angle_shift_copy)
            alpha_shift = alpha - referenced_image.shape[0]//2
            degree = alpha_shift * np.pi / referenced_image.shape[0] * 180 / np.pi
            # Assume rotation between frame is -5 ~ 5 deg
            if abs(degree) < 20:
                Flag = True
            else: angle_shift_copy[alpha] = -100000

        impulse_map = abs(np.fft.ifft(phase_shift))
        impulse_map -= impulse_map.min()
        impulse_map /= impulse_map.max()

        std_shift = (reference_theta_fft * reference_theta_fft.conjugate()) / (abs(reference_theta_fft) * abs(reference_theta_fft))
        std_impulse_map = abs(np.fft.ifft(std_shift))
        std_impulse_map -= std_impulse_map.min()
        std_impulse_map /= std_impulse_map.max()

        #print((0.9 < std_impulse_map).sum(), (0.8 < impulse_map).sum())


        # angle optimization
        alpha_shift, rot_err = optimize(reference_polar_theta, current_polar_theta, initial_guess=alpha_shift)
        rotation = alpha_shift * np.pi / referenced_image.shape[0] * 180 / np.pi

        #print("rot_err in radian: {}".format(rot_err * np.pi / referenced_image.shape[0]))


        return (rotation, 0, 0, np.var(impulse_map), np.var(std_impulse_map), rot_err * np.pi / referenced_image.shape[0])



    # Estimate horizontal translation
    rotated_referenced_image = rotate(referenced_image, angle=relativeAngle, reshape=False)
    rr_vertical_projection = np.sum(rotated_referenced_image, axis=0)
    cur_vertical_projection = np.sum(current_image, axis=0)
    #print(np.linspace(0, rr_vertical_projection.shape[0], rr_vertical_projection.shape[0]+1)[:-1])

    rr_fft = np.fft.fft(rr_vertical_projection)
    cur_fft = np.fft.fft(cur_vertical_projection)
    eps = cur_fft.max() * 1e-15
    phase_shift = (rr_fft * cur_fft.conjugate()) / (abs(rr_fft) * abs(cur_fft) + eps)
    horizontal_cps = abs(np.fft.ifft(phase_shift))
    horizontal_cps = np.fft.fftshift(horizontal_cps)

    Flag = False
    dx = None
    horizontal_cps_copy = horizontal_cps.copy()

    while not Flag:
        trans_x = np.argmax(horizontal_cps_copy)
        trans_x_shift = trans_x - rr_fft.shape[0]//2

        # Assume horizontal translation between frame is -5 ~ 5 pixel
        if abs(trans_x_shift) < 10:
            Flag = True
            dx = trans_x_shift
        else: horizontal_cps_copy[trans_x] = -100000

    # Estimate vertical translation
    rr_horizontal_projection = np.sum(rotated_referenced_image, axis=1)
    cur_horizontal_projection = np.sum(current_image, axis=1)
    rr_fft = np.fft.fft(rr_horizontal_projection)
    cur_fft = np.fft.fft(cur_horizontal_projection)
    eps = cur_fft.max() * 1e-15
    phase_shift = (rr_fft * cur_fft.conjugate()) / (abs(rr_fft) * abs(cur_fft) + eps)
    vertical_cps = abs(np.fft.ifft(phase_shift))
    vertical_cps = np.fft.fftshift(vertical_cps)

    Flag = False
    dy = None
    vertical_cps_copy = vertical_cps.copy()
    while not Flag:

        trans_y = np.argmax(vertical_cps_copy)
        trans_y_shift = trans_y - rr_fft.shape[0] // 2
        # Assume vertical translation between frame is -5 ~ 5 pixel
        if abs(trans_y_shift) < 5:
            Flag = True
            dy = trans_y_shift
        else:
            vertical_cps_copy[trans_y] = -100000



    # x, y shift optimization
    dx, x_error = optimize(rr_vertical_projection, cur_vertical_projection, initial_guess=dx)
    dy, y_error = optimize(rr_horizontal_projection, cur_horizontal_projection, initial_guess=dy)

    return (dx, dy*-1, x_error, y_error)







