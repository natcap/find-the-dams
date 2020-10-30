#!/bin/bash -x
# Authenticate the gcloud SDK so it can read from buckets
/usr/local/gcloud-sdk/google-cloud-sdk/bin/gcloud auth activate-service-account --key-file=natgeo-dams-1e56a9f3ab62.json
python dam_detector_server.py 2>&1
