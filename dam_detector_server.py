# coding=UTF-8
"""This server will manage a worker cloud."""
import argparse
import collections
import json
import logging
import os
import pathlib
import sqlite3
import subprocess
import sys
import threading
import time
import traceback

from flask import Flask
from osgeo import gdal
import flask
import retrying
import requests

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.INFO)

QUADS_TO_PROCESS_PATH = 'quads_to_process.gpkg'
LOGGER = logging.getLogger(__name__)

WORKSPACE_DIR = 'workspace'
DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'test_natgeo_dams_database_2020_07_01.db')

DAM_INFERENCE_WORKER_KEY = 'dam_inference_worker'
JOBS_PER_WORKER = 3

APP = Flask(__name__, static_url_path='', static_folder='')


@APP.route('/favicon.ico')
def favicon():
    """Return the favicon for webbrowser decoration."""
    return flask.send_from_directory(
        os.path.join(APP.root_path, 'images'), 'favicon.ico',
        mimetype='image/vnd.microsoft.icon')


@APP.route('/get_status/', methods=['POST'])
@retrying.retry(
    wait_exponential_multiplier=1000, wait_exponential_max=10000,
    stop_max_attempt_number=5)
def processing_status():
    """Return results about polygons that are processing."""
    try:
        return 'unimplemented'
    except Exception:
        LOGGER.exception('encountered exception')
        return traceback.format_exc()


@retrying.retry(wait_exponential_multiplier=100, wait_exponential_max=1000)
def _execute_sqlite(
        sqlite_command, database_path,
        argument_list=None, execute='execute', fetch=None):
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
    for dirname in [WORKSPACE_DIR]:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    # monitor for new or dead hosts
    client_monitor_thread = threading.Thread(
        target=client_monitor,
        args=(DAM_INFERENCE_WORKER_KEY,),
        )
        # kwargs={'local_hosts': {'localhost'}})
    client_monitor_thread.daemon = True
    client_monitor_thread.start()

    # schedule work
    work_manager_thread = threading.Thread(
        target=work_manager,
        args=(QUADS_TO_PROCESS_PATH,))
    work_manager_thread.daemon = True
    work_manager_thread.start()
    work_manager_thread.join()

    return
    APP.run(host='0.0.0.0', port=80)


class Worker(object):
    """Manages the state of a worker.

    API Functions:
        do_inference
        job_status
            'idle'
            'processing'
            'complete'
            'error'
        get_result

    API data keys:
        'quad_url'
        'dam_bounding_box_list'
            [(lng_min, lat_min, lng_max, lat_max), ...]
    '''

    """
    __next_id = 0
    __worker_set = dict()

    def __new__(cls, worker_ip, port=8080):
        """Construct new Worker or return reference to existing one.

        Args:
            cls (ClassType): Worker.
            worker_ip (str): IP address of worker. If this Worker has been
                created before it will reference that worker.
            port (int): which port to connect on

        Returns:
            New instance of Worker or in the case of existing IP a reference
            to an existing worker.
        """
        if worker_ip in Worker.__worker_set:
            return Worker.__worker_set[worker_ip]

        instance = super().__new__(cls)
        instance.worker_ip = worker_ip
        instance.port = port
        instance.active = False
        instance.id = Worker.__next_id
        Worker.__next_id += 1
        Worker.__worker_set[worker_ip] = instance
        return instance

    def __eq__(self, other):
        """Two workers are equal if they have the same id."""
        return self.worker_ip == other.worker_ip

    def __hash__(self):
        """Uniquely define a worker by its IP."""
        return hash(self.worker_ip)

    def __repr__(self):
        """Worker is uniquely identified by id, but IP is useful too."""
        return f'Worker: {self.worker_ip}({self.id})'

    def send_job(self, job_payload):
        """Send a job to the worker."""
        worker_rest_url = (
            f'http://{self.worker_ip}:{self.port}/do_inference')
        response = requests.post(
            worker_rest_url, json={'quad_uri': job_payload})
        if not response:
            raise RuntimeError(f'something went wrong {response.text}')
        self.active = True

    def get_status(self, job_payload):
        """Return 'idle', 'processing', 'complete', 'error'."""
        if not self.active:
            raise RuntimeError(
                f'Worker {self.worker_ip} tested but is not active.')
        worker_rest_url = (
            f'http://{self.worker_ip}:{self.port}/job_status')
        response = requests.post(
            worker_rest_url, json={'quad_uri': job_payload})
        return response.json()['status']

    def get_result(self, job_payload):
        """Return result if complete."""
        worker_rest_url = (
            f'http://{self.worker_ip}:{self.port}/get_result')
        response = requests.post(
            worker_rest_url, json={'quad_uri': job_payload})
        if response:
            LOGGER.debug(f'got {response.json()} for {self.job_payload}')
            return response.json()
        raise RuntimeError(f'bad response {response}')


def work_manager(quad_vector_path, update_interval=5.0):
    """Manager to record and schedule work.

    Args:
        quad_vector_path (str): path to vector containing quads with uri fields
            and processed fields. Do work on 'processed=0' fields then set
            to 1 when 'uri' is processed.

    Returns:
        None.
    """
    available_workers = set()
    worker_to_payload_list_map = collections.defaultdict(list)
    quad_uri_to_fid = {}

    # load quads to process to get fid & uri field
    unprocessed_fid_uri_list = []
    quad_vector = gdal.OpenEx(quad_vector_path, gdal.OF_VECTOR)
    quad_layer = quad_vector.GetLayer()
    quad_layer.SetAttributeFilter('processed=0')
    LOGGER.info('building work list')
    for quad_feature in quad_layer:
        fid = quad_feature.GetFID()
        quad_url = quad_feature.GetField('quad_uri')
        quad_uri = quad_url.replace('https://storage.googleapis.com/', 'gs://')
        unprocessed_fid_uri_list.append((fid, quad_uri))
        quad_uri_to_fid[quad_uri] = fid
    LOGGER.info(f'{len(unprocessed_fid_uri_list)} quads to process')

    try:
        while True:
            start_time = time.time()
            local_global_workers = list(GLOBAL_WORKERS)
            while local_global_workers:
                global_worker = local_global_workers.pop()
                if (global_worker not in available_workers and
                        len(worker_to_payload_list_map[global_worker]) <
                        JOBS_PER_WORKER):
                    LOGGER.debug(f'adding {global_worker} to available_workers')
                    available_workers.add(global_worker)

            # Schedule any available work to the workers
            while available_workers and unprocessed_fid_uri_list:
                free_worker = available_workers.pop()
                jobs_to_add = (
                    JOBS_PER_WORKER -
                    len(worker_to_payload_list_map[free_worker]))
                try:
                    for _ in range(jobs_to_add):
                        payload = unprocessed_fid_uri_list.pop()
                        LOGGER.debug(f'processing payload {payload}')
                        free_worker.send_job(payload[1])
                        worker_to_payload_list_map[free_worker].append(payload)
                except Exception:
                    LOGGER.exception(
                        f'unable to send job {payload[1]} to {free_worker}')
                    unprocessed_fid_uri_list.append(payload)

            # This loop checks if any of the workers are done, processes that
            # work and puts the free workers back on the free queue
            worker_to_payload_map_swap = collections.defaultdict(list)
            while worker_to_payload_list_map:
                scheduled_worker, payload_list = (
                    worker_to_payload_list_map.popitem())
                LOGGER.debug(f'*** scheduled worker payload list: {scheduled_worker} {payload_list}')
                while payload_list:
                    job_payload = payload_list.pop(0)
                    try:
                        status = scheduled_worker.get_status(job_payload[1])
                    except Exception:
                        LOGGER.exception(f'{scheduled_worker} failed')
                        status = 'error'
                    if status == 'error':
                        LOGGER.error(f'{scheduled_worker} failed on job {job_payload}')
                        unprocessed_fid_uri_list.append(job_payload)
                    elif status == 'complete':
                        # If job is complete, process result and put the
                        # free worker back in the free worker pool
                        payload = scheduled_worker.get_result(job_payload[1])
                        LOGGER.debug(f'{job_payload} COMPLETE result: {payload}')

                        LOGGER.info(
                            f"Update {DATABASE_PATH} With Completed Quad "
                            f"{payload['quad_uri']} "
                            f"{payload['dam_bounding_box_list']}")
                        if payload['dam_bounding_box_list']:
                            _execute_sqlite(
                                '''
                                INSERT INTO detected_dams
                                    (lng_min, lat_min, lng_max, lat_max,
                                     probability, country_list, image_uri)
                                VALUES(?, ?, ?, ?, -1, '', '');
                                ''', DATABASE_PATH,
                                argument_list=payload['dam_bounding_box_list'],
                                execute='many')

                        LOGGER.info(
                            f"Update Planet Quad Vector {payload['quad_uri']}")
                        quad_vector = gdal.OpenEx(
                            quad_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
                        quad_layer = quad_vector.GetLayer()
                        quad_layer.StartTransaction()
                        quad_feature = quad_layer.GetFeature(
                            quad_uri_to_fid[payload['quad_uri']])
                        quad_feature.SetField('processed', 1)
                        quad_layer.SetFeature(quad_feature)
                        quad_feature = None
                        quad_layer.CommitTransaction()
                        quad_layer = None
                        quad_vector = None
                    else:
                        LOGGER.info(
                            f'status for {scheduled_worker} is {status} '
                            f'({job_payload})')
                        worker_to_payload_map_swap[scheduled_worker].append(
                            job_payload)
            # swap back any remaining workers
            LOGGER.debug(f'swapping back {worker_to_payload_map_swap}')
            worker_to_payload_map = worker_to_payload_map_swap

            if (len(unprocessed_fid_uri_list) == 0 and
                    len(worker_to_payload_map) == 0):
                LOGGER.info('all done with work')
                return

            time.sleep(max(0, update_interval-(time.time()-start_time)))

    except Exception:
        LOGGER.exception('work manager failed')


def client_monitor(client_key, update_interval=5.0, local_hosts=None):
    """Watch for new clients and add them to the worker set.

    Args:
        client_key (str): filter on that key.
        update_interval (float): update interval to check for new instances

    Returns:
        Never.
    """
    try:
        while True:
            start_time = time.time()
            LOGGER.debug('checking for compute instances')
            result = subprocess.run(
                'gcloud compute instances list '
                f'--filter="metadata.items.key={client_key} AND status=RUNNING" '
                '--format=json', capture_output=True, shell=True).stdout
            live_workers = set()
            if local_hosts is not None:
                live_workers.update([Worker(host) for host in local_hosts])
            for instance in json.loads(result):
                network_ip = instance['networkInterfaces'][0][
                    'accessConfigs'][0]['natIP']
                LOGGER.debug(f"{instance['name']} {network_ip}")
                live_workers.add(Worker(network_ip))

            # rather than clear the set and reset it, we construct the set
            # by removing missing elements and adding new ones. this way we
            # don't get a glitch where a worker looks like it's missing because
            # the set is getting reset
            new_workers = live_workers - GLOBAL_WORKERS
            # Remove any clients that are missing
            GLOBAL_WORKERS.intersection_update(live_workers)
            # Add in any clients that are new
            LOGGER.debug(f'new workers: {new_workers}')
            GLOBAL_WORKERS.update(new_workers)
            LOGGER.debug(f'GLOBAL_WORKERS: {GLOBAL_WORKERS}')
            time.sleep(max(update_interval - (time.time() - start_time), 0))
    except Exception:
        LOGGER.exception('client monitor failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Start dam detection server.')
    args = parser.parse_args()
    GLOBAL_WORKERS = set()
    main()
