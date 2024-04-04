import time

import cv2
import numpy as np

cam = cv2.VideoCapture('./data/Lane_Detection_Test_Video_01.mp4')

screen_width = 1920
screen_height = 1080

frame_width = int(screen_width // 3)
frame_height = int(screen_height // 4)

upper_left = int(frame_width * 0.45), int(frame_height * 0.75)
upper_right = int(frame_width * 0.55), int(frame_height * 0.75)
lower_left = 0, int(frame_height)
lower_right = int(frame_width), int(frame_height)
trapezoid_bounds = [upper_left, upper_right, lower_right, lower_left]

blur_kernel_size = 5
blur_kernel = (blur_kernel_size, blur_kernel_size)

binary_threshold = 100

left_top = (0, 0)
left_bottom = (0, frame_height)
right_top = (0, 0)
right_bottom = (0, frame_height)

pow = 8

target_fps = 38
frame_duration = 1.0 / target_fps


def draw_final_lines(frame, left_line, right_line, original_frame):
    blank_frame_left = np.zeros_like(frame)
    blank_frame_right = np.zeros_like(frame)

    left_top, left_bottom, right_top, right_bottom = get_line_top_and_bottom(left_line, right_line)

    cv2.line(blank_frame_left, left_top, left_bottom, (255, 255, 255), 3)
    cv2.line(blank_frame_right, right_top, right_bottom, (255, 255, 255), 3)

    current_frame = np.float32([[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]])
    target_frame = np.float32(trapezoid_bounds)

    magic_matrix = cv2.getPerspectiveTransform(current_frame, target_frame)
    base_frame_left = cv2.warpPerspective(blank_frame_left, magic_matrix, (frame_width, frame_height))
    base_frame_right = cv2.warpPerspective(blank_frame_right, magic_matrix, (frame_width, frame_height))

    # cv2.imshow('Base Frame Left', base_frame_left)
    # cv2.imshow('Base Frame Right', base_frame_right)

    coordinates_left = np.argwhere(base_frame_left > 0)
    coordinates_right = np.argwhere(base_frame_right > 0)

    left_x = coordinates_left[:, 1]
    left_y = coordinates_left[:, 0]

    right_x = coordinates_right[:, 1]
    right_y = coordinates_right[:, 0]

    original_frame_copy = original_frame.copy()

    original_frame_copy[left_y, left_x] = [50, 50, 250]
    original_frame_copy[right_y, right_x] = [50, 250, 50]

    return original_frame_copy


def get_line_top_and_bottom(left_line, right_line):
    left_top_y = 0
    left_bottom_y = frame_height
    right_top_y = 0
    right_bottom_y = frame_height

    left_top_x = int((left_top_y - left_line[0]) / left_line[1])
    left_bottom_x = int((left_bottom_y - left_line[0]) / left_line[1])
    right_top_x = int((right_top_y - right_line[0]) / right_line[1])
    right_bottom_x = int((right_bottom_y - right_line[0]) / right_line[1])

    if -10 ** pow <= left_top_x <= 10 ** pow:
        left_top = (left_top_x, left_top_y)
    if -10 ** pow <= left_bottom_x <= 10 ** pow:
        left_bottom = (left_bottom_x, left_bottom_y)
    if -10 ** pow <= right_top_x <= 10 ** pow:
        right_top = (right_top_x, right_top_y)
    if -10 ** pow <= right_bottom_x <= 10 ** pow:
        right_bottom = (right_bottom_x, right_bottom_y)

    return left_top, left_bottom, right_top, right_bottom


def draw_lines(frame, left_line, right_line):
    left_top, left_bottom, right_top, right_bottom = get_line_top_and_bottom(left_line, right_line)

    cv2.line(frame, left_top, left_bottom, 200, 5)
    cv2.line(frame, right_top, right_bottom, 100, 5)

    middle_of_screen_x = frame_width // 2

    cv2.line(frame, (middle_of_screen_x, 0), (middle_of_screen_x, frame_height), 255, 1)

    return frame


def get_coordinates_of_street_markings(frame):
    noiseless_frame = frame.copy()
    noiseless_frame[:, :int(frame_width * 0.05)] = 0
    noiseless_frame[:, int(frame_width * 0.95):] = 0

    # cv2.imshow('Noiseless', noiseless_frame)

    coordinates_left = np.argwhere(noiseless_frame[:, :int(frame_width * 0.5)] > 0)
    coordinates_right = np.argwhere(noiseless_frame[:, int(frame_width * 0.5):] > 0)

    left_xs = coordinates_left[:, 1]
    left_ys = coordinates_left[:, 0]

    right_xs = coordinates_right[:, 1] + int(frame_width * 0.5)
    right_ys = coordinates_right[:, 0]

    return left_xs, left_ys, right_xs, right_ys


def binarize_frame(frame):
    _, binary_frame = cv2.threshold(frame, binary_threshold, 255, cv2.THRESH_BINARY)

    return binary_frame


def edge_detection_with_sobel_filter(frame):
    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)
    float_frame = np.float32(frame)

    vertical_edges = cv2.filter2D(float_frame, -1, sobel_vertical)
    horizontal_edges = cv2.filter2D(float_frame, -1, sobel_horizontal)

    # vertical_image = cv2.convertScaleAbs(vertical_edges)
    # horizontal_image = cv2.convertScaleAbs(horizontal_edges)
    # cv2.imshow('Vertical Edges', vertical_image)
    # cv2.imshow('Horizontal Edges', horizontal_image)

    combined_edges = np.sqrt(np.square(vertical_edges) + np.square(horizontal_edges))
    combined_image = cv2.convertScaleAbs(combined_edges)

    return combined_image


def top_down_perspective(frame):
    not_stretched = np.float32(trapezoid_bounds)
    stretched = np.float32([[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]])

    magic_matrix = cv2.getPerspectiveTransform(not_stretched, stretched)
    top_down_frame = cv2.warpPerspective(frame, magic_matrix, (frame_width, frame_height))

    return top_down_frame


def select_road(frame):
    black_frame = frame.copy()
    black_frame[:, :] = 0

    cv2.fillConvexPoly(black_frame, np.array(trapezoid_bounds), (1, 1, 1))
    road_frame = frame * black_frame

    # cv2.imshow('Trapezoid', black_frame * 255)

    return road_frame


def process_frame(frame):
    # Ex1: Resize the frame
    small_frame = cv2.resize(frame, (frame_width, frame_height))
    cv2.imshow('Small', small_frame)

    # Ex2: Convert the frame to gray
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray_frame)

    # Ex3: Select the road
    road_frame = select_road(gray_frame)
    cv2.imshow('Road', road_frame)

    # Ex4: Top-down perspective
    top_down_frame = top_down_perspective(road_frame)
    cv2.imshow('Top-down', top_down_frame)

    # Ex5: Blur the top-down frame
    blurred_frame = cv2.blur(top_down_frame, blur_kernel)
    cv2.imshow('Blur', blurred_frame)

    # Ex6: Edge detection with Sobel filter
    sobel_frame = edge_detection_with_sobel_filter(blurred_frame)
    cv2.imshow('Sobel', sobel_frame)

    # Ex7: Binarize the frame
    binary_frame = binarize_frame(sobel_frame)
    cv2.imshow('Binary', binary_frame)

    # Ex8: Get the coordinates of the street markings
    left_xs, left_ys, right_xs, right_ys = get_coordinates_of_street_markings(binary_frame)

    # Ex9: Find the lines that detect the edges of the lane and draw them
    left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, 1)
    right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, 1)
    lines_frame = draw_lines(binary_frame, left_line, right_line)
    cv2.imshow('Lines', lines_frame)

    # Ex10: Draw the final lines
    final_frame = draw_final_lines(small_frame, left_line, right_line, small_frame)
    cv2.imshow('Final Frame', final_frame)


if __name__ == '__main__':
    while True:
        start_time = time.time()

        ret, frame = cam.read()

        if ret is False:
            break

        process_frame(frame)

        elapsed_time = time.time() - start_time
        wait_time = max(0, frame_duration - elapsed_time)
        time.sleep(wait_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
