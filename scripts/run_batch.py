import os
import glob
from multiprocessing import Pool, cpu_count
from lake_bathymetry.core import process_file

if __name__ == '__main__':
    folder = r"your/path/to/tif/files"
    files = glob.glob(os.path.join(folder, '*.tif'))
    errors = []

    with Pool(cpu_count()) as pool:
        for fname, success, msg in pool.imap_unordered(process_file, files):
            if not success:
                print(f"{fname} failed: {msg}")
                errors.append((fname, msg))

    print(f" All done. Errors: {len(errors)}")
    if errors:
        with open("error_log.txt", "w") as f:
            for fname, msg in errors:
                f.write(f"{fname} -- {msg}\n")