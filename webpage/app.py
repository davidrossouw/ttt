import json
import os
import time
from flask import Flask, render_template, session, redirect, url_for
from flask_session import Session
from tempfile import mkdtemp

app = Flask(__name__)

app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


# Load ttt lookup table
TTT_MOVE_PATH = './ttt_minimax.json'
with open(TTT_MOVE_PATH, 'r') as f:
    ttt_moves = json.load(f)

def boardToString(board:list) -> str:
    flat_board = [col for row in board for col in row]
    board_string = ''.join([x if x is not None else '-' for x in flat_board])
    return board_string



def checkGame(board:list) -> None:
    '''
    
    '''
    #check rows
    for i in range(len(board)):
        if (board[i][0] == board[i][1] == board[i][2]):
            if(board[i][0] != None):
                session['game'] = board[i][0] + '_wins'
                return 
    #check cols
    for i in range(len(board)):
        if (board[0][i] == board[1][i] == board[2][i]):
            if(board[0][i] != None):
                session['game'] = board[0][i] + '_wins'
                return

    #check diagonals
    if(board[0][0] == board[1][1] == board[2][2]):
        if(board[0][0] != None):
            session['game'] = board[0][0] + '_wins'
            return
    #check diagonals
    if(board[2][0] == board[1][1] == board[0][2]):
        if(board[2][0] != None):
            session['game'] = board[2][0] + '_wins'
            return
    #check if game is in progress
    for i in range(len(board)):
        for j in range(len(board)):
            if(board[i][j] == None):
                session['game'] = 'in progress'
                return

    #game is drawn since there is no winner 
    #and all boxes are filled
    session['game'] = 'draw'
    return


@app.route("/")
def index():
    # Initial state
    if "board" not in session:
        session["board"] = [[None, None, None], [None, None, None], [None, None, None]]
        session["winner"] = False
        session["turn"] = "X"
        session["game"] = 'in progress'
        return render_template("game.html", board=session["board"], game=session["game"])

    
    ## Computer turn
    # Check game
    checkGame(session['board'])
    # Check if finished
    if session['game'] != 'in progress':
        return render_template("game.html", board=session["board"], game=session["game"])


    board_string = boardToString(session["board"])
    move = ttt_moves.get(board_string)
    if not move:
        session.pop('board')
        return redirect(url_for("index"))

    row = (move-1) // 3
    col = (move-1) % 3
    session["board"][row][col] = "O"

    # Check game
    checkGame(session['board'])
        
    return render_template("game.html", board=session["board"], game=session["game"])


@app.route("/play/<int:row>/<int:col>")
def play(row, col):
    session["board"][row][col] = session["turn"]
    return redirect(url_for("index"))


@app.route("/reset")
def reset():
    session.pop('board')
    return redirect(url_for("index"))


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=int(
        os.environ.get('PORT', 8080)), debug=False, use_reloader=False)

