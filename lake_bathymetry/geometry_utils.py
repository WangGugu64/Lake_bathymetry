import numpy as np
from skimage import draw
from scipy.stats import linregress

def EightNeighbor(p):
    return np.array([
        [p[0], p[1]-1], [p[0], p[1]+1], [p[0]-1, p[1]], [p[0]+1, p[1]],
        [p[0]-1, p[1]-1], [p[0]-1, p[1]+1], [p[0]+1, p[1]-1], [p[0]+1, p[1]+1]
    ])

def draw_extended_line(binaryimage, v1, v2):
    direction_vector = v2 - v1
    unit_v = direction_vector / np.linalg.norm(direction_vector)
    temp = v2.copy()
    while True:
        if binaryimage[int(temp[0]), int(temp[1])] == 0:
            temp -= unit_v
            break
        temp += unit_v
    r, c = draw.line(int(v2[0]), int(v2[1]), int(temp[0]), int(temp[1]))
    length = np.linalg.norm(np.array([r[0], c[0]]) - np.array([r[-1], c[-1]]))
    return length, unit_v



def TotalDistance(slope, binaryimage, point_inside, vector):
    temp1, temp2 = point_inside.copy(), point_inside.copy()
    while binaryimage[int(temp1[0]), int(temp1[1])] != 0:
        temp1 += vector
    while binaryimage[int(temp2[0]), int(temp2[1])] != 0:
        temp2 -= vector
    r, c = draw.line(int(temp2[0]), int(temp2[1]), int(temp1[0]), int(temp1[1]))
    Water_Array = np.array([[r[i], c[i]] for i in range(len(r))])
    length = len(Water_Array)
    p1 = np.int16(temp1 + vector)
    p2 = np.int16(temp2 - vector)
    s1, s2 = slope[p1[0], p1[1]], slope[p2[0], p2[1]]

    if s1 == 0 and s2 == 0:
        weight = 0.5
    else:
        if s1 > 3 * s2:
            s2 = s1
        elif s2 > 3 * s1:
            s1 = s2
        weight = s1 / (s1 + s2)
    current_len = max(1, int(weight * length))
    point_lowest = Water_Array[current_len - 1]
    return length, p1, p2, point_lowest

def circular_moving_average(data, window_size):
    extended = np.concatenate((data[-window_size + 1:], data, data[:window_size - 1]))
    smoothed = np.convolve(extended, np.ones(window_size) / window_size, mode='valid')
    return smoothed[:len(data)]
