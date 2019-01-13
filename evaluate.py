from game2048.game import Game
from game2048.displays import Display
import logging
from keras.models import load_model 

def single_run(logger, models, size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(logger=logger, models=models, game=game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score
# def single_run(size, score_to_win, AgentClass, **kwargs):
#     game = Game(size, score_to_win)
#     agent = AgentClass(game=game, display=Display(), **kwargs)
#     agent.play(verbose=True)
#     return game.score

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50
    logging.basicConfig(filename='test_3.log', format='%(asctime)s%(levelname)s%(message)s')
    logger = logging.getLogger('test_3')
    hdlr = logging.FileHandler('test_3.log')
    formatter = logging.Formatter('%(asctime)s%(levelname)s%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    '''====================
    Use your own agent here.'''
    from game2048.agents import XzxAgent as TestAgent
    '''===================='''
    # model = [load_model('xzx_rnn_model.h5'),load_model('xzx_rnn_model_2.h5')]
    models = [load_model('xzx_rnn_model_13.h5'), load_model('xzx_rnn_model_4.h5')] 
    scores = []
    for i in range(N_TESTS):
        score = single_run(logger, models, GAME_SIZE, SCORE_TO_WIN,
                            AgentClass=TestAgent)
        scores.append(score)
        info = "game: {} score: {}".format(i, score) 
        logger.info(info)
    score_dict = {}
    for score in scores:
        if score in score_dict.keys():
            score_dict[score] += 1
        else:
            score_dict[score] = 1
    print("score_dict {}".format(score_dict))
    info = "Ave_score: {}".format(sum(scores)/N_TESTS)
    logger.info(info)
    print(info)