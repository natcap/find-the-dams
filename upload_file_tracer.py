"""Tracer to upload file."""
import sys

import requests


file_path = sys.argv[1]
target_url = sys.argv[2]

print('uploading %s to %s' % (file_path, target_url))

r = requests.post(target_url)
print(r.json())
upload_url = r.json()['upload_url']

files = {'file': open(file_path, 'rb')}
r = requests.put(upload_url, files=files)
print(r)
