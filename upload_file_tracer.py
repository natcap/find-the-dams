"""Tracer to upload file."""
import numpy
import PIL.Image
from osgeo import gdal
import requests


file_path = r"D:\dams_debug\2377-1710.tif"

png_driver = gdal.GetDriverByName('PNG')
base_image = gdal.OpenEx(file_path, gdal.OF_RASTER)
png_path = 'image.png'
png_driver.CreateCopy(png_path, base_image)

image = PIL.Image.open(png_path).convert("RGB")
height, width = image.size
image_array = numpy.array(image.getdata()).reshape((height, width, 3))
print(image_array.shape)

target_url = "http://localhost:8080/api/v1/detect_dam"

print('uploading %s to %s' % (file_path, target_url))

r = requests.post(target_url)
print(r.json())
upload_url = r.json()['upload_url']

files = {'file': open(png_path, 'rb')}
r = requests.put(upload_url, files=files)
print(r)
