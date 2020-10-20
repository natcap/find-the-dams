# coding=UTF-8
"""This server will manage a worker cloud."""
import argparse
import datetime
import json
import logging
import os
import pathlib
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid

from flask import Flask
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import flask
import numpy
import PIL
import PIL.ImageDraw
import pygeoprocessing
import requests
import retrying
import shapely.geometry
import shapely.ops
import shapely.prepared
import shapely.strtree


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.INFO)

WORLD_BORDERS_VECTOR_PATH = 'world_borders.gpkg'
LOGGER = logging.getLogger(__name__)

WORKSPACE_DIR = 'workspace'
DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'natgeo_dams_database_2020_07_01.db')
REQUEST_TIMEOUT = 5
FRAGMENT_SIZE = (419, 419)  # dims to fragment a Planet quad
THRESHOLD_LEVEL = 0.08
QUERY_LIMIT = 20000  # how many dams to fetch at a time on the app
DETECTOR_POLL_TIME = 3  # seconds to wait between detector complete checks
GLOBAL_HOST_SET = None
GLOBAL_HOST_QUEUE = None
DATABASE_STATUS_STR = None
HOST_FILE_PATH = None
GLOBAL_LOCK = None
WORKING_GRID_ID_STATUS_MAP = None
FRAGMENT_ID_STATUS_MAP = None
SESSION_UUID = None
STATE_TO_COLOR = {
    'unscheduled': '#333333',
    'scheduled': '#FF3333',
    'downloaded': '#FF6600',
    'analyzing': '#6666FF',
    'complete': '#00FF33',
}

DAM_STATE_COLOR = {
    'identified': '#33FF00',
    'pre_known': '#0000F0',
}

APP = Flask(__name__, static_url_path='', static_folder='')


@APP.route('/favicon.ico')
def favicon():
    """Return the favicon for webbrowser decoration."""
    return flask.send_from_directory(
        os.path.join(APP.root_path, 'images'), 'favicon.ico',
        mimetype='image/vnd.microsoft.icon')


@APP.route('/')
def index():
    """Flask entry point."""
    return flask.render_template(
        'dashboard.html', **{
            'message': 'Stats will go here.',
        })


@APP.route('/processing_status/', methods=['POST'])
@retrying.retry(
    wait_exponential_multiplier=1000, wait_exponential_max=10000,
    stop_max_attempt_number=5)
def processing_status():
    """Return results about polygons that are processing."""
    try:
        last_known_dam_id = int(flask.request.form.get('last_known_dam_id'))
        payload = None
        polygons_to_update = {}
        LOGGER.debug(flask.request.form)
        database_uri = 'file:%s?mode=ro' % DATABASE_PATH
        connection = sqlite3.connect(database_uri, uri=True)
        cursor = connection.cursor()

        # fetch all grids
        cursor.execute(
            "SELECT grid_id, processing_state, lat_min, lng_min, lat_max, "
            "lng_max FROM work_status")

        # construct a return object that indicated which polygons should be
        # updated on the client

        polygons_to_update = {
            'grid_%s' % grid_id: {
                'bounds': [[lat_min, lng_min], [lat_max, lng_max]],
                'color': STATE_TO_COLOR[state],
                'fill': 'true' if state == 'unscheduled' else 'false',
                'weight': 1,
            } for (
                grid_id, state, lat_min, lng_min, lat_max, lng_max) in
            cursor.fetchall()
        }

        with GLOBAL_LOCK:
            for grid_id, fragment_info in FRAGMENT_ID_STATUS_MAP.items():
                polygons_to_update[grid_id] = fragment_info

        cursor.execute(
            "SELECT CAST(dam_id as INT) AS dam_id_n, pre_known, "
            "   lat_min, lng_min, lat_max, lng_max "
            "FROM detected_dams "
            "WHERE dam_id_n>? "
            "ORDER BY dam_id_n "
            "LIMIT %d" % QUERY_LIMIT, (last_known_dam_id,))
        dam_count = 0
        max_dam_id = last_known_dam_id
        for dam_id, pre_known, lat_min, lng_min, lat_max, lng_max in cursor:
            dam_count += 1
            max_dam_id = max(max_dam_id, int(dam_id))
            polygons_to_update[dam_id] = {
                'color': (
                    DAM_STATE_COLOR['pre_known'] if pre_known == int(1)
                    else DAM_STATE_COLOR['identified']),
                'bounds': [
                    [lat_min, lng_min],
                    [lat_max, lng_max]],
                'fill': 'false',
                'weight': 1,
            }
        payload = {
            'query_time': str(datetime.datetime.now()),
            'max_dam_id': max_dam_id,
            'polygons_to_update': polygons_to_update,
            'session_uuid': SESSION_UUID,
            'all_sent': 'true' if dam_count < QUERY_LIMIT else 'false',
        }
        cursor.close()
        connection.commit()
        return json.dumps(payload)
    except Exception:
        LOGGER.exception('encountered exception')
        return traceback.format_exc()


@retrying.retry(wait_exponential_multiplier=100, wait_exponential_max=1000)
def _execute_sqlite(
        sqlite_command, database_path,
        argument_list=None, mode='read_only', execute='execute', fetch=None):
    """Execute SQLite command and attempt retries on a failure.

    Args:
        sqlite_command (str): a well formatted SQLite command.
        database_path (str): path to the SQLite database to operate on.
        argument_list (list): `execute == 'execute` then this list is passed to
            the internal sqlite3 `execute` call.
        execute (str): must be either 'execute', 'many', or 'script'.
        fetch (str): if not `None` can be either 'all' or 'one'.
            If not None the result of a fetch will be returned by this
            function.

    Returns:
        result of fetch if `fetch` is not None.

    """
    cursor = None
    connection = None
    if argument_list is None:
        argument_list = []
    try:
        if fetch:
            ro_uri = r'%s?mode=ro' % pathlib.Path(
                os.path.abspath(database_path)).as_uri()
            LOGGER.debug(
                '%s exists: %s', ro_uri, os.path.exists(os.path.abspath(
                    database_path)))
            connection = sqlite3.connect(ro_uri, uri=True)
        else:
            connection = sqlite3.connect(database_path)

        if execute == 'execute':
            cursor = connection.execute(sqlite_command, argument_list)
        elif execute == 'many':
            cursor = connection.executemany(sqlite_command, argument_list)
        elif execute == 'script':
            cursor = connection.executescript(sqlite_command)
        else:
            raise ValueError('Unknown execute mode: %s' % execute)

        result = None
        payload = None
        if fetch == 'all':
            payload = (cursor.fetchall())
        elif fetch == 'one':
            payload = (cursor.fetchone())
        elif fetch is not None:
            raise ValueError('Unknown fetch mode: %s' % fetch)
        if payload is not None:
            result = list(payload)
        cursor.close()
        connection.commit()
        connection.close()
        return result
    except Exception:
        LOGGER.exception('Exception on _execute_sqlite: %s', sqlite_command)
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.commit()
            connection.close()
        raise


def main():
    """Entry point."""
    for dirname in [WORKSPACE_DIR,]:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    work_queue = queue.Queue()
    unprocessed_grids = _execute_sqlite(
        '''
        SELECT grid_id, lng_min, lat_min, lng_max, lat_max
        FROM work_status
        WHERE processed=0
        ''', DATABASE_PATH, fetch='all')

    for grid in unprocessed_grids:
        work_queue.put(grid)

    LOGGER.debug('grids sent to work queue')

    # TODO: monitor for new or dead hosts
    #   TODO: manage new hosts with a new thread

    client_monitor_thread = threading.Thread(
        target=client_monitor,
        args=('key', work_queue))
    client_monitor_thread.daemon = True
    client_monitor_thread.start()

    client_monitor_thread.join()

    return

    APP.run(host='0.0.0.0', port=80)


def client_monitor(client_key):
    """Watch for new clients and add them to the worker set.

    Args:
        client_key (str): filter on that key.

    Returns:
        Never.
    """
    worker_set = set()
    try:
        while True:
            LOGGER.debug('checking for compute instances')
            result = subprocess.run(
                'gcloud compute instances list --filter="labels=value" '
                '--format=json', capture_output=True, shell=True).stdout
            result_ip_set = set()
            for instance in json.loads(result):
                LOGGER.debug(instance)
                network_ip = instance['networkInterfaces'][0]['networkIP']
                LOGGER.debug(
                    f"{instance['name']} {network_ip}")
                result_ip_set.add(network_ip)
            new_ip_set = result_ip_set - worker_set
            removed_ip_set = worker_set - result_ip_set
            worker_set = result_ip_set
            LOGGER.debug(
                f'new ip set: {new_ip_set}, '
                f'removed_ip_set: {removed_ip_set}, '
                f'worker_set: {worker_set}')
            time.sleep(5)
    except Exception:
        LOGGER.exception('client monitor failed')


def host_file_monitor():
    """Watch inference_host_file_path & update inference_worker_host_queue.

    Parameters:
        inference_host_file_path (str): path to a file that contains lines
            of http://[host]:[port]<?label> that can be used to send inference
            work to. <label> can be used to use the same machine more than
            once.
        inference_worker_host_queue (queue.Queue): new hosts are queued here
            so they can be pulled by other workers later.
        inference_worker_work_queue


    """
    last_modified_time = 0
    worker_id = -1
    while True:
        try:
            current_modified_time = os.path.getmtime(inference_host_file_path)
            if current_modified_time != last_modified_time:
                last_modified_time = current_modified_time
                with open(inference_host_file_path, 'r') as ip_file:
                    ip_file_contents = ip_file.readlines()
                with GLOBAL_LOCK:
                    global GLOBAL_HOST_SET
                    old_host_set = GLOBAL_HOST_SET
                    GLOBAL_HOST_SET = set([
                        line.strip() for line in ip_file_contents
                        if line.startswith('http')])
                    new_hosts = GLOBAL_HOST_SET.difference(old_host_set)
                    for new_host in new_hosts:
                        inference_worker_host_queue.put(new_host)
                    n_hosts = len(new_hosts)
                for _ in range(n_hosts):
                    worker_id += 1
                    LOGGER.info(
                        'starting new inference_worker host %d', worker_id)
                    inference_worker_thread = threading.Thread(
                        target=inference_worker,
                        args=(inference_worker_work_queue,
                              inference_worker_host_queue,
                              DATABASE_PATH, worker_id))
                    inference_worker_thread.start()
            time.sleep(DETECTOR_POLL_TIME)
        except Exception:
            LOGGER.exception('exception in `host_file_monitor`')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Start dam detection server.')
    args = parser.parse_args()

    GLOBAL_LOCK = threading.Lock()
    GLOBAL_HOST_SET = set()
    GLOBAL_HOST_QUEUE = set()
    WORKING_GRID_ID_STATUS_MAP = {}
    FRAGMENT_ID_STATUS_MAP = {}
    SESSION_UUID = uuid.uuid4().hex
    main()
