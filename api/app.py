import datetime
import json
import logging
import os
import time

from auth import gcloud_auth
from flask import Flask, request, jsonify
from pytz import timezone

app = Flask(__name__)
logger = logging.getLogger('app')
logger.setLevel(20)

# Load ttt lookup table
TTT_MOVE_PATH = './ttt_minimax.json'
with open(TTT_MOVE_PATH, 'r') as f:
    ttt_moves = json.load(f)


@app.before_request
def authenticate_request():
    """Authenticates every request."""
    gcloud_auth(request.headers.get('Authorization'), logger)


@app.route('/', methods=['GET'])
def hello():
    """Return a friendly HTTP greeting."""
    who = request.args.get('who', 'there')
    # Write a log entry
    logger.info('who: %s', who)

    return f'Hello {who}!\n'


@app.route('/run', methods=['POST'])
def run():
    start = time.time()
    timestamp = datetime.datetime.now(
        timezone('America/Toronto')).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    data = request.get_json()
    board = data.get('board', False)
    if not board:
        return 'board data missing', 400
    
    move = ttt_moves.get(board)
    if not move:
        return 'board not found', 400

    end = time.time()
    execution_time = round(end - start, 2)
    logger.info(f"Success! Total execution time: {execution_time} sec.")
    logger.info(f"Move: {move}")

    return jsonify({'board': board, 'move': move}), 200


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=int(
        os.environ.get('PORT', 8080)), debug=False, use_reloader=True)
