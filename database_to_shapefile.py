"""Script to analyze how many dams found and how many pre-known."""
import logging
import os
import pathlib
import sqlite3
import sys

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import rtree
import shapely.wkb

TARGET_VECTOR_PATH = r"known_dams.gpkg"
BASE_DAMS_DB_PATH = "docker-image/workspace/test_natgeo_dams_database_2020_07_01.db"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)

def _execute_sqlite(
        sqlite_command, database_path, argument_list=None,
        mode='read_only', execute='execute', fetch=None):
    """Execute SQLite command and attempt retries on a failure.

    Parameters:
        sqlite_command (str): a well formatted SQLite command.
        database_path (str): path to the SQLite database to operate on.
        argument_list (list): `execute == 'execute` then this list is passed to
            the internal sqlite3 `execute` call.
        mode (str): must be either 'read_only' or 'modify'.
        execute (str): must be either 'execute', 'many', or 'script'.
        fetch (str): if not `None` can be either 'all' or 'one'.
            If not None the result of a fetch will be returned by this
            function.

    Returns:
        result of fetch if `fetch` is not None.

    """
    cursor = None
    connection = None
    try:
        if mode == 'read_only':
            ro_uri = r'%s?mode=ro' % pathlib.Path(
                os.path.abspath(database_path)).as_uri()
            LOGGER.debug(
                '%s exists: %s', ro_uri, os.path.exists(os.path.abspath(
                    database_path)))
            connection = sqlite3.connect(ro_uri, uri=True)
        elif mode == 'modify':
            connection = sqlite3.connect(database_path)
        else:
            raise ValueError('Unknown mode: %s' % mode)

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
    bounding_box_list = _execute_sqlite(
        f'''
        SELECT lng_min, lat_min, lng_max, lat_max
        FROM detected_dams
        WHERE probability=-1.0
        GROUP BY lng_min, lat_min, lng_max, lat_max
        ''', BASE_DAMS_DB_PATH, fetch='all', argument_list=[])
    gpkg_driver = ogr.GetDriverByName('GPKG')
    vector = gpkg_driver.CreateDataSource(f'detected_dams.gpkg')
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    layer = vector.CreateLayer(
        'detected_dams', wgs84_srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn('grid_id', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('processed', ogr.OFTString))

    LOGGER.info(f'starting transaction')
    layer.StartTransaction()
    for lng_min, lat_min, lng_max, lat_max in (
            bounding_box_list):
        if lng_min < -180:
            continue
        #LOGGER.debug(f'{lng_min}, {lat_min}, {lng_max}, {lat_max}')
        box = shapely.geometry.box(lng_min, lat_min, lng_max, lat_max)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkb(box.wkb))
        layer.CreateFeature(feature)

    LOGGER.info(f'commiting transaction')
    layer.CommitTransaction()


if __name__ == '__main__':
    main()
