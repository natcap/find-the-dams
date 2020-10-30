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
import numpy
import retrying
import requests

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
QUADS_TO_PROCESS_PATH = 'quads_to_process.gpkg'
LOGGER = logging.getLogger(__name__)

WORKSPACE_DIR = 'workspace'
DATABASE_PATH = os.path.join(
    WORKSPACE_DIR, 'test_natgeo_dams_database_2020_07_01.db')

DAM_INFERENCE_WORKER_KEY = 'dam_inference_worker'
JOBS_PER_WORKER = 3

UNPROCESSED_URI_LIST = None
START_COUNT = None
START_TIME = None

APP = Flask(__name__, static_url_path='', static_folder='')


@APP.route('/favicon.ico')
def favicon():
    """Return the favicon for webbrowser decoration."""
    return flask.send_from_directory(
        os.path.join(APP.root_path, 'images'), 'favicon.ico',
        mimetype='image/vnd.microsoft.icon')


@APP.route('/get_status', methods=['GET'])
def processing_status():
    """Return results about polygons that are processing."""
    if UNPROCESSED_URI_LIST is None:
        return 'booting up'
    left_to_process = len(UNPROCESSED_URI_LIST)
    current_time = time.time()
    global START_COUNT
    processing_rate = RATE_ESTIMATOR.get_rate()
    hours_left = left_to_process * processing_rate / 3600
    return (
        ('%d of %d quads left to process<br>' % (
            left_to_process, START_COUNT)) +
        ('%.2f%% complete<br>' % ((1-left_to_process/START_COUNT)*100.)) +
        ('%.4f hours processing<br>' % ((current_time-START_TIME) / 3600)) +
        ('processing %.2fs/dam<br>' % (processing_rate)) +
        ('estimate %.4f hours left<br>' % hours_left) +
        ('%d workers up') % len(GLOBAL_WORKERS))


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
#    work_manager_thread.join()
    APP.run(host='0.0.0.0', port=80)


class RateEstimator(object):
    def __init__(self, n_to_track):
        """Keep track of last `n_to_track` times."""
        self.n_to_track = n_to_track
        self._complete_time_list = []

    def complete(self):
        """Note another complete task."""
        self._complete_time_list.append(time.time())
        while len(self._complete_time_list) > self.n_to_track:
            self._complete_time_list.pop(0)

    def get_rate(self):
        """Return sec/completions."""
        if len(self._complete_time_list) < 2:
            return -99999
        time_span = (
            self._complete_time_list[-1] - self._complete_time_list[0])
        seconds_per_entry = time_span / len(self._complete_time_list)
        return seconds_per_entry


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
        return 'Worker: %s(%s)' % (self.worker_ip, self.id)

    def send_job(self, quad_uri_list):
        """Send a job to the worker.

        Args:
            quad_uri_list (list): a list of quad uris to process on this
            worker.

        Returns:
            None
        """
        worker_rest_url = (
            'http://%s:%s/do_inference' % (self.worker_ip, self.port))
        response = requests.post(
            worker_rest_url, json={'quad_uri_list': quad_uri_list})
        if not response:
            raise RuntimeError('something went wrong ' +response.text)
        self.active = True

    def get_status(self, quad_uri_list):
        """Return list of status or result.

        Args:
            quad_uri_list (list): list of quad uris to query.

        Return:
            a list of same length as ``quad_uri_list`` containing 'scheduled',
            'processing', 'error', or list of bounding boxes which are the
            result.
        """
        if not self.active:
            raise RuntimeError(
                'Worker %s tested but is not active.' % self.worker_ip)
        worker_rest_url = (
            'http://%s:%s/job_status' % (self.worker_ip, self.port))
        response = requests.post(
            worker_rest_url, json={'quad_uri_list': quad_uri_list})
        return response.json()['status_list']

    def health_check(self):
        """Test health status."""
        try:
            heath_rest_url = (
                'http://%s:%s/health_check' % (self.worker_ip, self.port))
            response = requests.get(heath_rest_url)
            if response.ok:
                LOGGER.info(
                    'health check for %s is okay %s' % (
                        self.worker_ip, response.text))
            else:
                LOGGER.error(
                    '***health check for %s is BAD %s' % (
                        self.worker_ip, response.text))
        except Exception:
            LOGGER.exception('error on %s when health_check' % self.worker_ip)


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
    global UNPROCESSED_URI_LIST
    UNPROCESSED_URI_LIST = []
    quad_vector = gdal.OpenEx(quad_vector_path, gdal.OF_VECTOR)
    quad_layer = quad_vector.GetLayer()
    quad_layer.SetAttributeFilter('processed=0')
    LOGGER.info('building work list')
    for quad_feature in quad_layer:
        fid = quad_feature.GetFID()
        quad_url = quad_feature.GetField('quad_uri')
        quad_uri = quad_url.replace('https://storage.googleapis.com/', 'gs://')
        UNPROCESSED_URI_LIST.append(quad_uri)
        quad_uri_to_fid[quad_uri] = fid
    global START_COUNT
    START_COUNT = len(UNPROCESSED_URI_LIST)
    global START_TIME
    START_TIME = time.time()
    LOGGER.info('%d quads to process' % START_COUNT)

    try:
        while True:
            start_time = time.time()
            # Build set of active workers
            local_global_workers = list(GLOBAL_WORKERS)
            while local_global_workers:
                global_worker = local_global_workers.pop()
                if (global_worker not in available_workers and
                        len(worker_to_payload_list_map[global_worker]) <
                        JOBS_PER_WORKER):
                    LOGGER.debug(
                        'adding %s to available_workers' % global_worker)
                    available_workers.add(global_worker)

            # Schedule any available work to any available workers
            while available_workers and UNPROCESSED_URI_LIST:
                free_worker = available_workers.pop()
                jobs_to_add = (
                    JOBS_PER_WORKER -
                    len(worker_to_payload_list_map[free_worker]))
                if jobs_to_add:
                    url_list = UNPROCESSED_URI_LIST[-jobs_to_add:]
                    try:
                        free_worker.send_job(url_list)
                        worker_to_payload_list_map[free_worker].extend(
                            url_list)
                        # remove the last `jobs_to_add` number of elements
                        UNPROCESSED_URI_LIST = (
                            UNPROCESSED_URI_LIST[:-jobs_to_add])
                    except Exception:
                        LOGGER.exception(
                            'unable to send job list %s to ' % (url_list)
                            + str(free_worker))

            # This loop checks if any of the workers are done, processes that
            # work and puts the free workers back on the free queue
            # if the worker has an error, invalidate all the work sent to it
            # and set it up to reschedule
            worker_to_payload_list_map_swap = collections.defaultdict(list)
            while worker_to_payload_list_map:
                complete_payload_bb_list = []
                still_processing_payload_list = []
                scheduled_worker, quad_uri_list = (
                    worker_to_payload_list_map.popitem())
                LOGGER.debug(
                    'about to check %s with %s' % (
                        scheduled_worker, quad_uri_list))
                # check the payload on that worker
                try:
                    status_list = scheduled_worker.get_status(quad_uri_list)
                    for status, quad_uri in zip(status_list, quad_uri_list):
                        if isinstance(status, list):
                            # quad_uri is complete this is the result!
                            LOGGER.info('%s complete with %d detections' % (
                                quad_uri, len(status)))
                            complete_payload_bb_list.append((quad_uri, status))
                        else:
                            LOGGER.info('%s status: %s' % (quad_uri, status))
                            still_processing_payload_list.append(quad_uri)
                except Exception:
                    # if exception, invalidate any of the work
                    LOGGER.exception('%s failed' % scheduled_worker)
                    UNPROCESSED_URI_LIST.extend(quad_uri_list)
                    still_processing_payload_list = []
                    complete_payload_bb_list = []

                # put any work back on the list that's still processing
                if still_processing_payload_list:
                    worker_to_payload_list_map_swap[scheduled_worker] = \
                        still_processing_payload_list

                # record any complete work
                if complete_payload_bb_list:
                    quad_vector = gdal.OpenEx(
                        quad_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
                    quad_layer = quad_vector.GetLayer()
                    quad_layer.StartTransaction()

                    for quad_uri, bounding_box_list in \
                            complete_payload_bb_list:
                        LOGGER.info(
                            f"Update {DATABASE_PATH} With Completed Quad "
                            f"{quad_uri} "
                            f"{bounding_box_list}")
                        RATE_ESTIMATOR.complete()
                        if bounding_box_list:
                            _execute_sqlite(
                                '''
                                INSERT INTO detected_dams
                                    (lng_min, lat_min, lng_max, lat_max,
                                     probability, country_list, image_uri)
                                VALUES(?, ?, ?, ?, -1, '', '');
                                ''', DATABASE_PATH,
                                argument_list=bounding_box_list,
                                execute='many')

                        LOGGER.info(
                            f"Update Planet Quad Vector {quad_uri}")
                        quad_feature = quad_layer.GetFeature(
                            quad_uri_to_fid[quad_uri])
                        quad_feature.SetField('processed', 1)
                        quad_layer.SetFeature(quad_feature)

                    quad_feature = None
                    quad_layer.CommitTransaction()
                    quad_layer = None
                    quad_vector = None
            # swap back any workers that are still processing
            worker_to_payload_list_map = worker_to_payload_list_map_swap
            LOGGER.debug(
                'done with iteration value: %s' % str(
                    worker_to_payload_list_map))
            if (len(UNPROCESSED_URI_LIST) == 0 and
                    len(worker_to_payload_list_map) == 0):
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
                '/usr/local/gcloud-sdk/google-cloud-sdk/bin/gcloud compute instances list '
                '--filter="metadata.items.key=%s ' % client_key +
                'AND status=RUNNING" --format=json',
                shell=True, stdout=subprocess.PIPE).stdout
            live_workers = set()
            if local_hosts is not None:
                live_workers.update([Worker(host) for host in local_hosts])
            for instance in json.loads(result):
                try:
                    network_ip = instance['networkInterfaces'][0][
                        'accessConfigs'][0]['natIP']
                except Exception:
                    network_ip = instance['networkInterfaces'][0]['networkIP']
                live_workers.add(Worker(network_ip))

            # rather than clear the set and reset it, we construct the set
            # by removing missing elements and adding new ones. this way we
            # don't get a glitch where a worker looks like it's missing because
            # the set is getting reset
            new_workers = live_workers - GLOBAL_WORKERS
            # Remove any clients that are missing
            GLOBAL_WORKERS.intersection_update(live_workers)
            # Add in any clients that are new
            if new_workers:
                LOGGER.debug('new workers: %s' % str(new_workers))
            GLOBAL_WORKERS.update(new_workers)
            LOGGER.debug('GLOBAL_WORKERS: %s' % str(GLOBAL_WORKERS))
            time.sleep(max(update_interval - (time.time() - start_time), 0))
    except Exception:
        LOGGER.exception('client monitor failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Start dam detection server.')
    args = parser.parse_args()
    GLOBAL_WORKERS = set()
    RATE_ESTIMATOR = RateEstimator(30)
    main()
