import numpy as np

def Volume_Depth(file_name, origin_Lbm, dem, origindem, output_img, cellsize, cal_method):
    region_mask = output_img == 1
    dem_region = dem[region_mask]
    if len(dem_region) > 0:
        q5 = np.percentile(dem_region, 5)
        dem[region_mask] = np.where(dem[region_mask] < q5, q5, dem[region_mask])

    elevation_img_output = dem.copy()
    output_img[origin_Lbm == 1] = origindem[origin_Lbm == 1] - dem[origin_Lbm == 1]
    output_img[output_img < 0] = 0
    output_img[origin_Lbm == 0] = np.nan

    Volcul_img = output_img[origin_Lbm == 1] * cellsize[0] * cellsize[1]
    Volume = np.sum(Volcul_img)
    max_depth = np.max(output_img[origin_Lbm == 1])
    average_depth = np.mean(output_img[origin_Lbm == 1])
    file_name = file_name.split('.')[0]
    data = [file_name, average_depth, max_depth, Volume, cal_method]

    return output_img, elevation_img_output, data