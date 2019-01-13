import json
import numpy as np
from game2048.game import Game
from keras.models import load_model 


def generate_fingerprint(AgentClass, models,**kwargs):
    with open("board_cases.json") as f:
        board_json = json.load(f)

    game = Game(size=4, enable_rewrite_board=True)
    agent = AgentClass(models = models,game=game ,**kwargs)

    trace = []
    for board in board_json:
        game.board = np.array(board)
        direction = agent.step()
        trace.append(direction)
    fingerprint = "".join(str(i) for i in trace)
    return fingerprint


if __name__ == '__main__':
    from collections import Counter

    '''====================
    Use your own agent here.'''
    from game2048.agents import XzxAgent as TestAgent
    '''===================='''
    models = [load_model('xzx_rnn_model_4.h5')] 
    fingerprint = generate_fingerprint(TestAgent,models)

    with open("EE369_fingerprint.json", 'w') as f:        
        pack = dict()
        pack['fingerprint'] = fingerprint
        pack['statstics'] = dict(Counter(fingerprint))
        f.write(json.dumps(pack, indent=4))
