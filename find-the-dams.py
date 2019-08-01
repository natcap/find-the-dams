# coding=UTF-8
"""Module to process remote dam detection imagery."""
import json
import ast
import urllib
import sqlite3
import os
import sys
import logging

import taskgraph
from flask import Flask
import flask

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.INFO)

DAM_BOUNDING_BOX_URL = (
    'https://storage.googleapis.com/natcap-natgeo-dam-ecoshards/dams_database_md5_7acdf64cd03791126a61478e121c4772.db')
LOGGER = logging.getLogger(__name__)

WORKSPACE_DIR = 'workspace'
DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'find-the-dams.db')
N_WORKERS = -1

APP = Flask(__name__, static_url_path='', static_folder='')
APP.config['SECRET_KEY'] = b'\xe2\xa9\xd2\x82\xd5r\xef\xdb\xffK\x97\xcfM\xa2WH'


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


def initalize_database(database_path):
    """Create the data tracking database if it doesn't exist."""
    create_database_sql = (
        """
        CREATE TABLE IF NOT EXISTS spatial_analysis_units (
            id INTEGER NOT NULL PRIMARY KEY,
            state TEXT NOT NULL,
            lat_min REAL NOT NULL,
            lng_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lng_max REAL NOT NULL);

        CREATE INDEX IF NOT EXISTS sa_lat_min_idx
        ON spatial_analysis_units (lat_min);
        CREATE INDEX IF NOT EXISTS sa_lng_min_idx
        ON spatial_analysis_units (lng_min);
        CREATE INDEX IF NOT EXISTS sa_lat_max_idx
        ON spatial_analysis_units (lat_max);
        CREATE INDEX IF NOT EXISTS sa_lng_max_idx
        ON spatial_analysis_units (lng_max);
        CREATE UNIQUE INDEX IF NOT EXISTS sa_id_idx
        ON spatial_analysis_units (id);

        CREATE TABLE IF NOT EXISTS identified_dams (
            dam_id INTEGER NOT NULL PRIMARY KEY,
            pre_known INTEGER NOT NULL,
            dam_description TEXT NOT NULL,
            lat_min REAL NOT NULL,
            lng_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lng_max REAL NOT NULL);

        CREATE INDEX IF NOT EXISTS id_lat_min_idx
        ON identified_dams (lat_min);
        CREATE INDEX IF NOT EXISTS id_lng_min_idx
        ON identified_dams (lng_min);
        CREATE INDEX IF NOT EXISTS id_lat_max_idx
        ON identified_dams (lat_max);
        CREATE INDEX IF NOT EXISTS id_lng_max_idx
        ON identified_dams (lng_max);
        CREATE UNIQUE INDEX IF NOT EXISTS id_dam_id_idx
        ON identified_dams (dam_id);
        """)
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.executescript(create_database_sql)
    cursor.close()
    connection.commit()


def initalize_spatial_search_units(database_path):
    """Set the initial spatial search units in the database."""
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, N_WORKERS, 30.0)
    dam_bounding_box_bb_path = os.path.join(
        WORKSPACE_DIR, os.path.basename(DAM_BOUNDING_BOX_URL))
    task_graph.add_task(
        func=urllib.request.urlretrieve,
        args=(DAM_BOUNDING_BOX_URL, dam_bounding_box_bb_path),
        target_path_list=[dam_bounding_box_bb_path],
        task_name='download dam bb')
    # get all the original data processed
    task_graph.join()

    try:
        connection = sqlite3.connect(dam_bounding_box_bb_path)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT "
            "bounding_box_bounds, metadata, validation_table.key, "
            "database_id, description "
            "FROM validation_table "
            "INNER JOIN base_table on base_table.key = validation_table.key;")
        spatial_analysis_unit_list = []
        for payload in cursor:
            (bounding_box_bounds_raw, metadata_raw,
             key, database_id, description) = payload
            metadata = json.loads(metadata_raw)
            if 'checkbox_values' in metadata and (
                    any(metadata['checkbox_values'].values())):
                continue
            if 'comments' in metadata and (
                    'dry' in metadata['comments'].lower()):
                continue
            bounding_box_dict = ast.literal_eval(bounding_box_bounds_raw)
            if bounding_box_dict is None:
                continue
            lat_min, lat_max = list(sorted([
                x['lat'] for x in bounding_box_dict]))
            lng_min, lng_max = list(sorted([
                x['lng'] for x in bounding_box_dict]))
            spatial_analysis_unit_list.append(
                (key, 1, '%s:%s' % (database_id, description), lat_min,
                 lng_min, lat_max, lng_max))
    except Exception:
        LOGGER.exception("Exception encountered.")
    finally:
        LOGGER.info("all done, signaling stop")
        cursor.close()
        connection.commit()

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.executemany(
        'INSERT INTO identified_dams( '
        'dam_id, pre_known, dam_description, lat_min, lng_min, lat_max, '
        'lng_max) VALUES(?, ?, ?, ?, ?, ?, ?)', spatial_analysis_unit_list)
    cursor.close()
    connection.commit()


def main():
    """Entry point."""
    try:
        os.makedirs(WORKSPACE_DIR)
    except OSError:
        pass
    initalize_database(DATABASE_PATH)
    initalize_spatial_search_units(DATABASE_PATH)
    APP.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
