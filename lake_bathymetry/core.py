# === lake_bathymetry/core.py ===

import os
import numpy as np
import rasterio
from skimage import measure
from .analysis import *
from .io_utils import save_raster
from .volume import Volume_Depth
from .geometry_utils import EightNeighbor, draw_extended_line, TotalDistance

def process_file(tif):
    try:
        file_name = os.path.basename(tif)
        output_path = r"your/path/to/output/directory"
        os.makedirs(output_path, exist_ok=True)
        output_path1 = os.path.join(output_path, file_name.split('.')[0] + '.tif')

        with rasterio.open(tif) as src:
            Dem = src.read(1)
            Slope = src.read(2)
            transform = src.transform
            crs = src.crs
            x_res, y_res = src.res

        CellSize = [x_res, y_res]
        buffer_distances = np.floor_divide(np.array([100, 200, 300, 400, 500, 600]), CellSize[0]).astype(int)

        binary_image = water_get(Slope).astype(np.uint8)
        LBM = LakeSure(binary_image)
        H = Dem[Dem.shape[0] // 2, Dem.shape[1] // 2]
        OriginDem = Dem.copy()
        Real_Lbm = LBM.copy()
        Origin_LBM = LBM.copy()
        Dem[LBM == 1] = -1

        contours = measure.find_contours(LBM, 0)
        if not contours:
            return (file_name, False, 'No contours found')

        contour_points_land = np.int16(np.concatenate(contours))
        slope = np.zeros_like(Dem, dtype='float32')

        slope_buffer_mean_input, Slope_buffer, index = LakeParameter(LBM, Dem, Slope, buffer_distances)
        slope_buffer_mean = np.tan(np.radians(slope_buffer_mean_input))
        buffer_len = buffer_distances[index]

        for contour in contours:
            FirstSlopeCal(Dem, H, Origin_LBM, np.int16(contour), slope, slope_buffer_mean, CellSize, buffer_len)

        array_slope = np.array([slope[p[0], p[1]] for p in contour_points_land])

        method = 2
        if slope_buffer_mean < np.tan(np.radians(5)):
            method = 1

        Slope_stddev, mean = Data_stretching(Slope, slope, Slope_buffer, slope_buffer_mean)
        slope = Slope_stddev if mean < slope_buffer_mean else Data_stretching_1(slope)

        H_stander = H
        while np.any(LBM == 1):
            contour_points_water = []
            for pt in contour_points_land:
                for nb in EightNeighbor(pt):
                    if LBM[nb[0], nb[1]] == 1:
                        LBM[nb[0], nb[1]] = 0
                        contour_points_water.append(nb)
            contour_points_water = np.array(contour_points_water)
            LBM[contour_points_water[:, 0], contour_points_water[:, 1]] = 1

            for pt in contour_points_water:
                h_array = []
                length_array = []
                for nb in EightNeighbor(pt):
                    if LBM[nb[0], nb[1]] == 0:
                        current_length, vector = draw_extended_line(LBM, nb, pt)
                        cellsize_value = CellSize[0] if vector[0] == 0 else CellSize[1]
                        if np.linalg.norm(vector) > 1:
                            cellsize_value = np.linalg.norm(CellSize)
                        Origin_Length, p1, p2, plow = TotalDistance(slope, Origin_LBM, pt, vector)
                        h = calculate(slope, pt, Dem[nb[0], nb[1]], cellsize_value, p1, p2, plow, mean, current_length, method)
                        h_array.append(h)
                        length_array.append(Origin_Length)
                weights = np.reciprocal(length_array)
                weights /= np.sum(weights)
                Dem[pt[0], pt[1]] = np.dot(h_array, weights)
                LBM[pt[0], pt[1]] = 0

            contours = measure.find_contours(LBM, 0)
            if not contours:
                break
            contour_points_land = np.int16(np.concatenate(contours))

            if np.any(Dem[contour_points_land[:, 0], contour_points_land[:, 1]] > H_stander):
                mask = Dem[contour_points_land[:, 0], contour_points_land[:, 1]] > H_stander
                contour_points_land = contour_points_land[mask]
            else:
                H_stander = np.min(Dem[contour_points_land[:, 0], contour_points_land[:, 1]])

        Output_lake, Output_dem, Data = Volume_Depth(file_name, Real_Lbm, Dem, OriginDem, np.zeros_like(Dem, dtype='float32'), CellSize, method)
        save_raster(output_path1.replace('.tif', '_lake.tif'), Output_lake, crs, transform)
        save_raster(output_path1.replace('.tif', '_dem.tif'), Output_dem, crs, transform)

        print(f"✔️ {file_name} saved")
        return (file_name, True, None)
    except Exception as e:
        return (file_name, False, str(e))