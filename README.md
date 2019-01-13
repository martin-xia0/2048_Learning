# 2048_Learning
Agent for game 2048 based on leaning from a Strong Agent(imitation learning)

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`xzx_model.py`](game2048/xzx_model.py): the core of XzxAgent
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`evaluate.py`] : evaluate the agent
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`test.log`] : log for Agent evaluation
* [`get_data.py`] : get raw data from ExpectMax
* [`get_data.log`] : log for data mining
* [`preprocess_rnn.py`] : proprocess raw data to training data
* [`train_xzx_rnn.py`] : model definition and training
* [`train_model.log`] : log for model training


# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask



# To run the web app
```bash
python3 webapp.py
```

# To run the evaluation
```bash
python3 evaluation.py
```
It will return the average score of 50 tests

# LICENSE
The code is under Apache-2.0 License.
