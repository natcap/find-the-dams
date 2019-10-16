"""Tracer code to demo recieving an uploaded file."""
import uuid
import shutil
import threading
import os

import flask
from flask import Flask


APP = Flask(__name__, static_url_path='', static_folder='')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'od_workspace')

try:
    shutil.rmtree(UPLOAD_FOLDER)
except OSError:
    pass
try:
    os.makedirs(UPLOAD_FOLDER)
except OSError:
    pass
# Configure Flask app and the logo upload folder
APP.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
APP_PORT = 8080
#APP.config['SERVER_NAME'] = 'localhost:%s' % APP_PORT

# map session ids to current state
# can be
#  * 'waiting for upload'
#  * 'processing'
#  * 'error: <msg>'
#  * <url to download file>
SESSION_MANAGER_MAP = {}


@APP.route('/api/v1/detect_dam', methods=['POST'])
def detect_dam_init():
    """Initialize a new dam classifying image."""
    session_id = str(uuid.uuid4())
    print(session_id)
    with SESSION_MANAGER_LOCK:
        SESSION_MANAGER_MAP[session_id] = 'waiting for upload'
    return {
        'upload_url': flask.url_for(
            'detect_dam', _external=True, session_id=session_id)
    }


@APP.route('/api/v1/detect_dam/<string:session_id>', methods=['PUT'])
def detect_dam(session_id):
    """Flask entry point."""
    if session_id not in SESSION_MANAGER_MAP:
        return ('%s not a valid session', 400)
    with SESSION_MANAGER_LOCK:
        if SESSION_MANAGER_MAP[session_id] != 'waiting for upload':
            return (
                '%s in state %s' % session_id,
                SESSION_MANAGER_MAP[session_id], 400)
    print(flask.request.files['file'])
    target_path = os.path.join(UPLOAD_FOLDER, session_id)
    flask.request.files['file'].save(target_path)
    with SESSION_MANAGER_LOCK:
        SESSION_MANAGER_MAP[session_id] = 'processing'
    return "200"


if __name__ == '__main__':
    print(APP.root_path)
    SESSION_MANAGER_LOCK = threading.Lock()
    APP.run(host='0.0.0.0', port=APP_PORT)
