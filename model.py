import random
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout

ALL_MOVES = [[0,0],[0,1],[1,0],[1,1]]


class Move():
    def __init__(self, player:int, move):
        self.move = move
        self.player = player # 0 or 1
        self.non_player = 1 - player # 0 or 1
        assert move in ALL_MOVES  
        

class RandomPlayer():
    def __init__(self, name, idx=None):
        self.name = name
        self.idx = idx
    
    def select_move(self, board) -> Move:
        moves_available = board.moves_available(self.idx)
        move = random.choice(moves_available)
        return Move(self.idx, move)

    
class MiniMaxPlayer():
    def __init__(self, name, idx=None, max_depth=10):
        self.name = name
        self.idx = idx
        self.max_depth = max_depth
        
    def search(self, board) -> DiGraph:
        G = search(board, max_depth=self.max_depth)
        return G

    def select_move(self, board) -> Move:
        G = self.search(board)
        G, move = minimax(G)
        if not move:
            moves_available = board.moves_available(self.idx)
            move = random.choice(moves_available)
        return Move(self.idx, move)

        
class Board():
    def __init__(self, player1:tuple, player2:tuple):
        self._state = [player1, player2]
        
    @property
    def state(self):
        """Get the current state of the board."""
        return self._state
    

    def moves_available(self, player:int=0) -> list: # make this have a Board as input
        # [[0,0],[0,1],[1,0],[1,1]] # 'll', 'lr', 'rl', 'rr'
        moves = []
        active = player
        inactive = 1-player
        
        # player1 left hand not 0 and player2 left hand not 0
        if self.state[active][0] != 0 and self.state[inactive][0] != 0:
            moves.append([0,0])
            
        # player1 left hand not 0 and player2 left hand not 0
        if self.state[active][0] != 0 and self.state[inactive][1] != 0:
            moves.append([0,1])
            
        # player1 right hand not 0 and player2 left hand not 0
        if self.state[active][1] != 0 and self.state[inactive][0] != 0:
            moves.append([1,0])
            
        # player1 right hand not 0 and player2 left hand not 0
        if self.state[active][1] != 0 and self.state[inactive][1] != 0:
            moves.append([1,1])

        return moves

    @property
    def display(self):
        return str(self.state[0][0])+str(self.state[0][1])+ \
               str(self.state[1][0])+str(self.state[1][1])
    
    @property
    def is_winner(self):
        # player1 left hand == 0 and player1 right hand == 0
        if self.state[0][0] == 0 and self.state[0][1] == 0:
            return True
            
        # player2 left hand == 0 and player2 right hand == 0
        if self.state[1][0] == 0 and self.state[1][1] == 0:
            return True
        
        return False
    
    def update_board(self, move:Move):
        new_state = list(map(list, self.state))
        # Apply move
        new_state[move.non_player][move.move[1]] += self.state[move.player][move.move[0]]
        # If any hand > 4, set to zero (out of the game)
        if new_state[move.non_player][move.move[1]] > 4:
            new_state[move.non_player][move.move[1]] = 0

        new_board = Board(new_state[0], new_state[1])
        return new_board



    
def search(board:Board, max_depth=3) -> DiGraph:
    """
    Run game simulations from current game state to a maximum number
    of moves ahead (max_depth)
    Return the graph of possible moves and outcomes
    state = [(1,1), (1,1)]
    Assume first player is play to be maximized
    """

    n = 0 # node label which also serves as a node counter
    depth = 0
    
    G = nx.DiGraph()
    G.add_node(0, winner=None, player=0, board=board.state, board_p = board.display)
    
    # First branch in look ahead
    newleavelist=[]
    parent_node = n
    parent_board = Board(G.nodes[n]['board'][0], G.nodes[n]['board'][1])

    for move in ALL_MOVES:
        moves_available = parent_board.moves_available(player=0)
        if move not in moves_available:
            continue
        
        # Do move
        new_board = parent_board.update_board(Move(player=0, move=move))
        
        # Add move node to graph
        n=n+1
        G.add_node(n, winner=new_board.is_winner, player=1, board=new_board.state, board_p = new_board.display)
        G.add_edge(parent_node, n, move=move)
        if new_board.is_winner:
            continue
        newleavelist.append(n)
    
    depth=1
    # subsequent branches
    while depth < max_depth:
        leavelist = newleavelist[:]
        newleavelist = []
        for leave in leavelist: 
            # Get parent board
            parent_board = Board(G.nodes[leave]['board'][0], G.nodes[leave]['board'][1])
            for move in ALL_MOVES:
                moves_available = parent_board.moves_available(player=depth%2)
                if move not in moves_available:
                    continue
                # Do move
                new_board = parent_board.update_board(Move(player=depth%2, move=move))
                # Add move node to graph
                n=n+1
                G.add_node(n, winner=new_board.is_winner, player=1-depth%2, 
                           board=new_board.state, board_p=new_board.display)
                G.add_edge(leave, n, move=move)
                if new_board.is_winner:
                    continue
                    
                newleavelist.append(n)
        depth=depth+1
    return G    


def minimax(G):
    """Perform minimax from node n on a NetworkX graph G.
    Assume node n is a maximiser node.
    Return node corresponding to best move
    """

    maxplayer = True
    minplayer = False
    def _minimax(G, n, player):

        # Base case, winning node found
        if G.nodes[n]["winner"]:
            if player == maxplayer:
                # min player won (with previous move)
                score = -100
            elif player == minplayer:
                # max player won (with previous move)
                score = 100
            else:
                assert True == False
                
            G.nodes[n].update({'score': score})
            return score

        if player == maxplayer:
            bestv = -1
            for child in G.successors(n):
                v = _minimax(G, child, minplayer)
                G.nodes[child].update({'score': v})
                bestv = max(bestv, v)
        else:
            bestv = 1
            for child in G.successors(n):
                v = _minimax(G, child, maxplayer)
                G.nodes[child].update({'score': v})
                bestv = min(bestv, v)
        return bestv

    # Find the best first move from the given node
    # Assume given node n is a maximiser node.
    best_move = None
    bestv = -1

    for child in G.successors(0):
        v = _minimax(G, child, minplayer)
        G.nodes[child].update({'score': v})

        if v > bestv:
            best_move = child
            bestv = v

    if best_move:
        return G, list(G.in_edges(best_move, data=True))[0][2]['move']
    else:
        return G, None



def draw_graph(G, fig_size = (5,5), node_label=None, edge_label=None):
    f, ax = plt.subplots(figsize=fig_size)
    G.graph.setdefault('graph', {})['rankdir'] = 'LR'
    # color nodes based on winner
    node_color = []
    node_size = 50
    for node in G.nodes(data=True):
        if node[1]['winner']:
            if node[1]['player'] == 1:
                node_color.append('green')
            if node[1]['player'] == 0:
                node_color.append('red')
        else:
            node_color.append('lightgray')
    pos = graphviz_layout(G, prog='dot')
    if node_label:
        nx.draw_networkx(G, pos=pos, labels=nx.get_node_attributes(G, node_label), node_color=node_color, node_size=node_size)
    else:
        nx.draw_networkx(G, pos=pos, with_labels=False, node_color=node_color, node_size=node_size)   
    if edge_label:
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, edge_label))
        
    plt.show()
    return

