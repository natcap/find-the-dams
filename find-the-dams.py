# coding=UTF-8
"""Module to process remote dam detection imagery."""
import datetime
import json
import ast
import urllib
import sqlite3
import os
import sys
import logging

import taskgraph
from osgeo import gdal
import shapely.prepared
import shapely.ops
import shapely.wkb
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
    'https://storage.googleapis.com/natcap-natgeo-dam-ecoshards/'
    'dams_database_md5_7acdf64cd03791126a61478e121c4772.db')
GLOBAL_POLYGON_URL = (
    'https://storage.googleapis.com/natcap-natgeo-dam-ecoshards/'
    'global_polygon_valid_md5_d1d71564442d850c8b1884e79e519d8f.gpkg')
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


@APP.route('/processing_status/', methods=['POST'])
def processing_status():
    """Return results about polygons that are processing."""
    LOGGER.debug('processing_status')
    result = json.dumps({
        'data': str(datetime.datetime.now()),
        })
    return result


def initalize_spatial_search_units(database_path, complete_token_path):
    """Set the initial spatial search units in the database.

    Parameters:
        database_path (str): path to SQLite database that's created by this
            call.
        complete_token_path (str): path to a file that's written if the
            entire initialization process has completed successfully.

    Returns:
        None.

    """
    create_database_sql = (
        """
        CREATE TABLE spatial_analysis_units (
            id INTEGER NOT NULL PRIMARY KEY,
            state TEXT NOT NULL,
            lat_min REAL NOT NULL,
            lng_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lng_max REAL NOT NULL);

        CREATE INDEX sa_lat_min_idx
        ON spatial_analysis_units (lat_min);
        CREATE INDEX sa_lng_min_idx
        ON spatial_analysis_units (lng_min);
        CREATE INDEX sa_lat_max_idx
        ON spatial_analysis_units (lat_max);
        CREATE INDEX sa_lng_max_idx
        ON spatial_analysis_units (lng_max);
        CREATE UNIQUE INDEX sa_id_idx
        ON spatial_analysis_units (id);

        CREATE TABLE identified_dams (
            dam_id INTEGER NOT NULL PRIMARY KEY,
            pre_known INTEGER NOT NULL,
            dam_description TEXT NOT NULL,
            lat_min REAL NOT NULL,
            lng_min REAL NOT NULL,
            lat_max REAL NOT NULL,
            lng_max REAL NOT NULL);

        CREATE INDEX id_lat_min_idx
        ON identified_dams (lat_min);
        CREATE INDEX id_lng_min_idx
        ON identified_dams (lng_min);
        CREATE INDEX id_lat_max_idx
        ON identified_dams (lat_max);
        CREATE INDEX id_lng_max_idx
        ON identified_dams (lng_max);
        CREATE UNIQUE INDEX id_dam_id_idx
        ON identified_dams (dam_id);
        """)
    LOGGER.debug('create database tables and indexes')
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.executescript(create_database_sql)
    cursor.close()
    connection.commit()

    dam_bounding_box_bb_path = os.path.join(
        WORKSPACE_DIR, os.path.basename(DAM_BOUNDING_BOX_URL))
    LOGGER.debug("download validated dam bounding box database")
    urllib.request.urlretrieve(DAM_BOUNDING_BOX_URL, dam_bounding_box_bb_path)
    LOGGER.debug("parse dam bounding box database for valid dams")
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

    LOGGER.debug("insert valid validated dams into `identified_dams` table")
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.executemany(
        'INSERT INTO identified_dams( '
        'dam_id, pre_known, dam_description, lat_min, lng_min, lat_max, '
        'lng_max) VALUES(?, ?, ?, ?, ?, ?, ?)', spatial_analysis_unit_list)
    cursor.close()
    connection.commit()

    # define the spatial search units
    LOGGER.debug("download download global polygon")
    global_polygon_path = os.path.join(
        WORKSPACE_DIR, os.path.basename(GLOBAL_POLYGON_URL))
    urllib.request.urlretrieve(GLOBAL_POLYGON_URL, global_polygon_path)

    LOGGER.debug("convert global polygon to shapely geometry")
    global_polygon_vector = gdal.OpenEx(global_polygon_path, gdal.OF_VECTOR)
    global_polygon_layer = global_polygon_vector.GetLayer()
    global_polygon_list = [
        shapely.wkb.loads(feature.GetGeometryRef().ExportToWkb())
        for feature in global_polygon_layer]
    LOGGER.debug("unary union global polygon geometry")
    global_polygon = shapely.ops.unary_union(global_polygon_list)
    LOGGER.debug("build spatial index")
    global_polygon_prep = shapely.prepared.prep(global_polygon)

    spatial_analysis_unit_list = []
    spatial_analysis_id = 0
    # 84N to 56S lat because that's what I eyed as the continent coverage
    for lat in range(-56, 85):
        LOGGER.debug('processing lat %d', lat)
        for lng in range(-180, 180):
            grid_geom = shapely.geometry.box(lng, lat, lng+1, lat+1)
            if global_polygon_prep.intersects(grid_geom):
                spatial_analysis_unit_list.append(
                    (spatial_analysis_id, 'unscheduled',
                     lat, lng, lat+1, lng+1))

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.executemany(
        'INSERT INTO spatial_analysis_units( '
        'id, state, lat_min, lng_min, lat_max, lng_max) '
        'VALUES(?, ?, ?, ?, ?, ?)', spatial_analysis_unit_list)
    cursor.close()
    connection.commit()

    with open(complete_token_path, 'w') as token_file:
        token_file.write(str(datetime.datetime.now()))


def main():
    """Entry point."""
    try:
        os.makedirs(WORKSPACE_DIR)
    except OSError:
        pass
    initalize_token_path = os.path.join(
        WORKSPACE_DIR, 'initalize_spatial_search_units.COMPLETE')
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, 0, 30.0)
    task_graph.add_task(
        func=initalize_spatial_search_units,
        args=(DATABASE_PATH, initalize_token_path),
        target_path_list=[initalize_token_path],
        task_name='initialize database')
    APP.run(host='0.0.0.0', port=8080)
    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()
