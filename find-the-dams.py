# coding=UTF-8
"""Module to process remote dam detection imagery."""
import sqlite3
import os
import sys
import logging

from flask import Flask
import flask

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('taskgraph').setLevel(logging.INFO)

LOGGER = logging.getLogger(__name__)

WORKSPACE_dir = 'workspace'
DATABASE_PATH = os.path.join(WORKSPACE_dir, 'find-the-dams.db')

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

        CREATE UNIQUE INDEX IF NOT EXISTS sa_lat_min_idx
        ON spatial_analysis_units (lat_min);
        CREATE UNIQUE INDEX IF NOT EXISTS sa_lng_min_idx
        ON spatial_analysis_units (lng_min);
        CREATE UNIQUE INDEX IF NOT EXISTS sa_lat_max_idx
        ON spatial_analysis_units (lat_max);
        CREATE UNIQUE INDEX IF NOT EXISTS sa_lng_max_idx
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

        CREATE UNIQUE INDEX IF NOT EXISTS id_lat_min_idx
        ON identified_dams (lat_min);
        CREATE UNIQUE INDEX IF NOT EXISTS id_lng_min_idx
        ON identified_dams (lng_min);
        CREATE UNIQUE INDEX IF NOT EXISTS id_lat_max_idx
        ON identified_dams (lat_max);
        CREATE UNIQUE INDEX IF NOT EXISTS id_lng_max_idx
        ON identified_dams (lng_max);
        CREATE UNIQUE INDEX IF NOT EXISTS id_dam_id_idx
        ON identified_dams (dam_id);
        """)
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.executescript(create_database_sql)
    cursor.close()
    connection.commit()


def main():
    """Entry point."""
    try:
        os.makedirs(WORKSPACE_dir)
    except OSError:
        pass
    initalize_database(DATABASE_PATH)
    APP.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
