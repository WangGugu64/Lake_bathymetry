import numpy as np
import rasterio
from openpyxl import Workbook

def save_raster(path, array, crs, transform):
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=array.shape[0], width=array.shape[1],
        count=1, dtype='float32',
        crs=crs, transform=transform, nodata=np.nan
    ) as dst:
        dst.write(array, 1)

def add_data_to_excel(data_rows, file_path):
    wb = Workbook()
    ws = wb.active
    ws.append(['Name', 'Ave_dep', 'Max_dep', 'Volume', 'Method'])
    for row in data_rows:
        ws.append(row)
    wb.save(file_path)