"""Tracer to upload file."""
import os
import sys
import shutil

import time
import numpy
import PIL.Image
from osgeo import gdal
import requests


file_path = sys.argv[2]

png_driver = gdal.GetDriverByName('PNG')
base_image = gdal.OpenEx(file_path, gdal.OF_RASTER)
png_path = 'image.png'
png_driver.CreateCopy(png_path, base_image)

image = PIL.Image.open(png_path).convert("RGB")
height, width = image.size
image_array = numpy.array(image.getdata()).reshape((height, width, 3))
print(image_array.shape)

host_and_port = "http://%s" % sys.argv[1]
detect_dam_url = "%s/api/v1/detect_dam" % host_and_port

print('uploading %s to %s' % (file_path, detect_dam_url))

r = requests.post(detect_dam_url)
print(r.json())
upload_url = r.json()['upload_url']

files = {'file': open(png_path, 'rb')}
r = requests.put(upload_url, files=files)
print(r.json())
status_url = r.json()['status_url']

while True:
    r = requests.get(status_url)
    okay = r.ok
    if okay:
        r = r.json()
        print(r)
        if r['status'] != 'complete':
            time.sleep(3)
        else:
            break
    else:
        print('error: %s', r.json()['status'])
        break

if okay:
    print(r['bounding_box_list'])
    download_url = r['annotated_png_url']
    local_path = os.path.basename(download_url)
    with requests.get(download_url, stream=True) as r:
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
print('done!')
