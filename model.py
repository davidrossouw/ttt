import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


class RandomPlayer():
    """
    A player that selects an available move at random
    """
    def __init__(self, marker):
        self.marker = marker
        
    def move(self, board):
        moves_available = board.moves_available(board.state)
        move = random.choice(moves_available)
        return move
    

class MiniMaxPlayer():
    """
    A player that searches all possible moves on the board
    up to search_depth, then uses the minimax algorithm to
    determine the best move to play.
    """

    def __init__(self, marker, search_depth=3):
        self.marker = marker
        self.search_depth = search_depth
        
    def move(self, board):
        """Move selection"""
        G = self.search(board=board, player=self, max_depth=self.search_depth)
        move = self.minimax(G)
        return move
    
    @staticmethod
    def search(board, player, max_depth): # -> DiGraph
        """
        Run game simulations from current game state to a maximum number
        of moves ahead (max_depth)
        Return the graph of possible moves and outcomes
        state = [(1,1), (1,1)]
        Assume player is player to be maximized
        """
        player_marker = player.marker
        opponent_marker = {'X': 'O', 'O': 'X'}.get(player.marker)

        turn_marker = player.marker
        non_turn_maker = opponent_marker

        depth = 0
        n = 0 # node label which also serves as a node counter

        G = nx.DiGraph()
        G.add_node(0, finished=board.is_finished(board.state), player=turn_marker, state=board.state)

        # First branch in look ahead
        newleavelist=[]

        for move in board.moves_available(state=board.state):   
            # Do move   
            new_state = board.play_move(state=board.state, move=move, marker=turn_marker)
            # Check if game is finished after move
            is_finished = board.is_finished(new_state)

            # Add move node to graph
            n=n+1
            G.add_node(n, finished=is_finished, player=non_turn_maker, state=new_state)
            G.add_edge(0, n, move=move)
            if is_finished:
                G.nodes[n]['result'] = {player_marker:'won', opponent_marker:'lost', 'draw':'draw'}.get(is_finished)
                continue
            newleavelist.append(n)

        depth+=1
        # Subsequent branches
        while depth < max_depth:
            # switch turns
            turn_marker, non_turn_maker = non_turn_maker, turn_marker
            leavelist = newleavelist[:]
            newleavelist = []
            for leave in leavelist:
                # Get parent state
                parent_state = G.nodes(data=True)[leave]['state'] #G.nodes(data=True)[list(G.pred[leave])[0]]['state']
                for move in board.moves_available(parent_state):
                    # Do move   
                    new_state = board.play_move(parent_state, move=move, marker=turn_marker)
                    # Check if game is finished after move
                    is_finished = board.is_finished(new_state)
                    # Add move node to graph
                    n=n+1
                    G.add_node(n, finished=is_finished, player=non_turn_maker, state=new_state)
                    G.add_edge(leave, n, move=move)
                    if is_finished:
                        G.nodes[n]['result'] = {player_marker: 'won', opponent_marker: 'lost', 'draw': 'draw'}.get(is_finished)
                        continue  
                    newleavelist.append(n)
            depth=depth+1
        return G

    
    @staticmethod
    def minimax(G):
        """
        Perform minimax from node n on a NetworkX graph G.
        Assume node n is a maximiser node.
        Return best move
        """
        maxplayer = True
        minplayer = False
        # Recursive tree search
        def _minimax(G, n, player):

            # Base case, winning node found
            if G.nodes[n]['finished']:
                if G.nodes[n]['result'] == 'won':
                    score = 100
                elif G.nodes[n]['result'] == 'lost':
                    score = -100
                elif G.nodes[n]['result'] == 'draw':
                    score = 0
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
        best_node = None
        bestv = -1

        for child in G.successors(0):
            v = _minimax(G, child, minplayer)
            G.nodes[child].update({'score': v})

            if v > bestv:
                best_node = child
                bestv = v

        if best_node:
            return G.pred[best_node][0]['move']

        else:
            scores = [(n[0], n[1]['score']) for n in G.subgraph(G.successors(0)).nodes(data=True)]
            random.shuffle(scores)
            best_node = max(scores, key = lambda t: t[1])[0]
            return G.pred[best_node][0]['move']
    
    @staticmethod    
    def draw_graph(G, fig_size = (5,5), node_label=None, edge_label=None):
        """A utility method to draw a given graph"""
        f, ax = plt.subplots(figsize=fig_size)
        G.graph.setdefault('graph', {})['rankdir'] = 'LR'
        # color nodes based on winner
        node_color = []
        node_size = 50
        for node in G.nodes(data=True):
            if node[1]['finished']:
                if node[1]['result'] == 'won':
                    node_color.append('green')
                elif node[1]['result'] == 'lost':
                    node_color.append('red')
                elif node[1]['result'] == 'draw':
                    node_color.append('yellow')
                else:
                    node_color.append('blue')
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
    
    

class Board():
    def __init__(self, state):
        self._state = state
        
    @property
    def state(self):
        """Get the current state of the board."""
        return self._state
    
    @staticmethod
    def moves_available(state):
        """Return the available moves on the board"""
        return list(zip(*np.where(state==0)))
    
    
    def update_board(self, move, marker):
        """Update the internal state of the board after a move"""
        marker_to_value = {'-': 0, 'X': 1, 'O': 2}
        # fail if move is not available
        assert self.state[move[0]][move[1]] == 0
        assert marker in ['X', 'O']
        self.state[move[0]][move[1]] = marker_to_value[marker]
        
    @staticmethod
    def draw(state):
        """Print the board"""
        size = state.shape[0]
        value_to_marker = {0: '-', 1: 'X', 2: 'O'}
        for m in range(size):
            print('|'.join([value_to_marker[state[m][n]] for n in range(size)]))
            if m != size-1:
                print('-'*((size*2)-1))
    
    @staticmethod
    def play_move(state, move, marker):
        """Return the new state of the board after a move"""
        new_state = np.copy(state)
        marker_to_value = {'-': 0, 'X': 1, 'O': 2}
        # fail if move is not available
        assert new_state[move[0]][move[1]] == 0
        assert marker in ['X', 'O']
        new_state[move[0]][move[1]] = marker_to_value[marker]
        return new_state
    
    
    @staticmethod    
    def is_finished(state):
        """
        Check of board is in a winning state
        Return None if not a winning state, otherwise
        Return the marker of the winning player (1 or 2) or 'draw' if draw
        """
        value_to_marker = {0: '-', 1: 'X', 2: 'O'}
        size = state.shape[0]
        rows = [state[i,:] for i in range(size)]
        cols = [state[:,j] for j in range(size)]
        diag = [np.array([state[i,i] for i in range(size)])]
        cross_diag = [np.array([state[(size-1)-i,i] for i in range(size)])]
        lanes = np.concatenate((rows, cols, diag, cross_diag))
        for lane in lanes:
            if set(lane) == {1}:
                #print(f"player {value_to_marker[1]} wins!")
                return value_to_marker[1]
            if set(lane) == {2}:
                #print(f"player {value_to_marker[2]} wins!")
                return value_to_marker[2]
        
        # check for draw
        if np.all(state!=0):
            #print('Draw!')
            return 'draw'
        
        # game still in progress
        return None
