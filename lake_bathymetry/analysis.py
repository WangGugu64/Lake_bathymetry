import numpy as np
import cv2
from scipy.stats import genextreme, skew, linregress
from shapely.geometry import Polygon
from shapely.validation import make_valid
from .geometry_utils import circular_moving_average
from skimage import measure, draw
def water_get(slope):
    return np.int8(np.where(slope < 0.02, 1, 0))

def EightNeighbor(p):
    left = [p[0], p[1] - 1]
    right = [p[0], p[1] + 1]
    up = [p[0] - 1, p[1]]
    down = [p[0] + 1, p[1]]
    Left_Up = [p[0] - 1, p[1] - 1]
    Right_Up = [p[0] - 1, p[1] + 1]
    Left_Down = [p[0] + 1, p[1] - 1]
    Right_Down = [p[0] + 1, p[1] + 1]
    return np.array([left, right, up, down, Left_Up, Right_Up, Left_Down, Right_Down])


def find_parabola_coefficients(k, length, m, x):  ### 求解抛物线 ###

    A = m / (2 * length)
    k = 2 * A * x
    # z = A_sol * x ** 2 + B_sol * x + C_sol
    # k_x = 2 * A_sol * x + B_sol
    # print(A, x, k)
    return k
def calculate(slope, point_now, dem_h, cellsize, point1, point2, pointlow, mean_slope, current_len,
              a):  # 记得加入方向判断(不需要了)
    point_slope_1 = slope[point1[0], point1[1]]
    point_slope_2 = slope[point2[0], point2[1]]
    total_length = np.linalg.norm(point1 - point2)
    X = np.linalg.norm(point_now - pointlow)
    x1 = np.linalg.norm(pointlow - point1)
    x2 = np.linalg.norm(pointlow - point2)

    if point_slope_1 > (3 * point_slope_2):
        point_slope_1 = point_slope_1 * 0.5
        point_slope_2 = point_slope_1  #######
    elif point_slope_2 > (3 * point_slope_1):
        point_slope_2 = point_slope_2 * 0.5
        point_slope_1 = point_slope_2  ######

        ############## 6.19晚加（新计算）二次函数
    if (min(point1[0], pointlow[0]) <= point_now[0] <= max(point1[0], pointlow[0]) and min(point1[1], pointlow[1]) <=
            point_now[1] <= max(point1[1], pointlow[1])):
        point_slope = point_slope_1
        oneside_length = x1
    else:
        point_slope = point_slope_2
        oneside_length = x2
    # if point_slope > slope_break[2]:
    #     point_slope = slope_break[2] / 3
    slope_now = find_parabola_coefficients(total_length, oneside_length, point_slope, X)
    # slope_now = slope_now * (np.sin(current_len / total_length * np.pi / 2))
    ''''''''
    slope_now = slope_now * (current_len / total_length)
    #slope_now = slope_now * ((current_len / total_length) ** a)
    # slope_now = slope_now * ((X / oneside_length) ** a)
    ''''''''
    h_now = dem_h - slope_now * cellsize
    # if (min(point1[0], pointlow[0]) <= point_now[0] <= max(point1[0], pointlow[0]) and min(point1[1], pointlow[1]) <=point_now[1] <= max(point1[1], pointlow[1])):
    # if (min(point1[0], pointlow[0]) <= point_now[0] <= max(point1[0], pointlow[0]) and min(point1[1], pointlow[1]) <=
    #         point_now[1] <= max(point1[1], pointlow[1])):
    #     length = np.linalg.norm(pointlow - point1)
    #     t = np.linalg.norm(point_now - point1)
    #     point_slope = point_slope_1
    # else:
    #     length = np.linalg.norm(pointlow - point2)
    #     t = np.linalg.norm(point_now - point2)
    #     point_slope = point_slope_2
    # slope_now = (point_slope / length) * (length - t)
    # slope_now = slope_now * (current_len / total_length)
    # h_now = dem_h - slope_now * cellsize
    ##############
    return h_now

def draw_slope_line(Slope, h, dem, binaryimage, v1, extend_length, cellsize):  # 函数重载（slope中的用法）
    # 计算点1到点2的方向向量
    # slope_this_point = 0
    eight_neighbor = EightNeighbor(v1)
    array_temp = np.array([])
    slope_this_point = 1 / ((cellsize[0] + cellsize[1])/2)
    for neighbor in eight_neighbor:
        if binaryimage[neighbor[0], neighbor[1]] == 1:
            vector = v1 - neighbor
            magnitude = np.linalg.norm(vector)
            if vector[0] == 0:
                cell_size = cellsize[0]
            else:
                cell_size = cellsize[1]
            if magnitude > 1:
                cell_size = np.linalg.norm(cell_size)
            stop_point = v1 + vector * extend_length
            # temp_stop_point = v1 + vector
            r, c = draw.line(int(stop_point[0]), int(stop_point[1]), int(v1[0]), int(v1[1]))

            # Land_slope_Array = np.array([[r[i], c[i]] for i in range(min(len(r), len(c)))])
            # point_first = Land_slope_Array[0]
            # point_last = Land_slope_Array[-1]
            Land_dem = np.array([dem[r[i], c[i]] for i in range(min(len(r), len(c)))])
            # slope_array = np.zeros(len(Land_slope_Array))
            slope_array = np.array([])
            #slope_this_point = 0

            if np.all(binaryimage[r[:], c[:]] == 0):

                # slope_this_point = (dem[point_first[0], point_first[1]] - dem[point_last[0], point_last[1]]) / (
                #             cellsize * extend_length)
                # # slope_this_point = np.mean(np.tan(np.radians(slope_S[r[:], c[:]])))


                # for i in range(len(r)-1):
                #     t = (dem[r[i], c[i]] - dem[r[-1], c[-1]]) / (cellsize * (len(r)-i))

                #     if t > 0:
                #         slope_array = np.append(slope_array, t)
                #
                # if slope_array.size != 0:
                #     q4 = np.percentile(slope_array, 40)
                #     q6 = np.percentile(slope_array, 60)
                #     filtered_slope_array = [x for x in slope_array if q4 <= x <= q6]
                #     if not filtered_slope_array:
                #         filtered_slope_array = (q4 + q6) / 2
                #     #slope_array[slope_array > q4] = qq2
                #     slope_this_point = np.mean(filtered_slope_array)
                #     if np.isnan(slope_this_point) == 1:
                #         print('nan')
                #######

                distances = np.arange(len(Land_dem)) * cell_size
                slope, intercept, r_value, p_value, std_err = linregress(distances, Land_dem)
                slope_this_point = -slope
                # 8.20加
                # print(slope)
                if slope_this_point <= 0:
                    slope_this_point = 1 / ((cellsize[0] + cellsize[1]) / 2)
                # slope_array = list(reversed(slope_array[1:]))
                # weight_factor = np.zeros_like(slope_array)
                # for i in range(len(slope_array)):
                #     weight_factor[i] = 1 / (i + 1)**2
                # weight_factor_sum = np.sum(weight_factor)
                # #weight_factor = weight_factor / weight_factor_sum
                # slope_this_point = (slope_array @ weight_factor) / weight_factor_sum
                # slope_this_point = np.mean(np.tan(np.radians(Slope[r[:], c[:]])))
            if slope_this_point != 0:
                array_temp = np.append(array_temp, slope_this_point)
    # q1 = np.percentile(array_temp, 10)
    # q2 = np.percentile(array_temp, 90)
    # array_temp[array_temp > q2] = q2
    # array_temp[array_temp < q1] = q1
    if array_temp.size != 0:
        slope_this_point = np.mean(array_temp)
    else:
        slope_this_point = 0

    return slope_this_point

def LakeSure(img):
    height, width = img.shape
    center_y, center_x = height // 2, width // 2
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(img)
    distances = [
        np.linalg.norm((centroid_x - center_x, centroid_y - center_y))
        for centroid_x, centroid_y in centroids[1:]
    ]
    closest_label = np.argmin(distances) + 1
    lake_mask = np.zeros_like(img)
    lake_mask[labels == closest_label] = 1
    return lake_mask

def Data_stretching(Slope, array_data, buffer_data, slope_buffer_data):
    non_zero_data = array_data[array_data != 0]
    min_val = np.min(non_zero_data)
    max_val = np.max(non_zero_data)
    buffer_data = Slope[buffer_data == 1]
    new_min = np.percentile(buffer_data, 5)
    new_max = np.percentile(buffer_data, 95)
    scaled_data = ((non_zero_data - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    scaled_data = np.tan(np.radians(scaled_data))
    result_data = np.copy(array_data)
    result_data[array_data != 0] = scaled_data
    result_data[result_data > np.percentile(scaled_data, 90)] = np.percentile(scaled_data, 90)
    return result_data, np.mean(non_zero_data)

def Data_stretching_1(array_data):
    non_zero_data = array_data[array_data != 0]
    min_v = np.min(non_zero_data)
    max_v = np.max(non_zero_data)
    new_min = np.percentile(non_zero_data, 0)
    new_max = np.percentile(non_zero_data, 95)
    scaled_data = ((non_zero_data - min_v) / (max_v - min_v)) * (new_max - new_min) + new_min
    result_data = np.copy(array_data)
    result_data[array_data != 0] = scaled_data
    return result_data

def LakeParameter(lbm, dem, slope, buffer_distances):
    contours, _ = cv2.findContours(np.uint8(lbm), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda x: cv2.contourArea(x))
    points = contour.reshape(-1, 2)

    if len(points) < 3:
        x, y = points[0]
        points = np.array([[x, y], [x + 1, y], [x, y + 1], [x, y]])
    polygon = make_valid(Polygon(points))

    if polygon.is_empty:
        raise ValueError("Invalid polygon")

    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    buffer_image = np.zeros_like(lbm, dtype=np.uint8)
    buffer_results = []

    for i, dist in enumerate(buffer_distances):
        buffer_polygon = polygon.buffer(dist)
        buffered_coords = np.array(buffer_polygon.exterior.coords, dtype=np.int32)
        current_buffer_image = np.zeros_like(lbm, dtype=np.uint8)
        cv2.fillPoly(current_buffer_image, [buffered_coords], 1)

        buffer_diff = current_buffer_image - buffer_image if i != 0 else current_buffer_image - lbm
        buffer_image = current_buffer_image
        buffer_values = dem[buffer_diff == 1].ravel()

        filtered_values = buffer_values[buffer_values <= np.percentile(buffer_values, 95)]
        elevation_mean = np.mean(filtered_values)
        buffer_results.append(elevation_mean)
        if i != 0 and buffer_results[i] - buffer_results[i - 1] <= 0:
            break
        temp_buffer = buffer_diff

    slope_values = slope[temp_buffer == 1].ravel()
    filtered_slope_values = slope_values[slope_values <= np.percentile(slope_values, 95)]
    slope_mean = np.mean(filtered_slope_values)
    return slope_mean, buffer_diff, i - 1

def FirstSlopeCal(dem, h, origin_lbm, contour1, slope, slope_buffer_mean, cellsize, buffer_len):
    array_temp = np.array([])

    for pt in contour1:
        slope[pt[0], pt[1]] = draw_slope_line(slope, h, dem, origin_lbm, pt, buffer_len, cellsize)
        array_temp = np.append(array_temp, slope[pt[0], pt[1]])

    array_temp[array_temp == 0] = slope_buffer_mean
    slope_temp = slope.copy()
    window_size = 3 if len(contour1) < 3 else min(3, len(contour1) // 2)
    smoothed = circular_moving_average(array_temp, window_size)
    for i in range(len(contour1)):
        slope[contour1[i][0], contour1[i][1]] = smoothed[i]