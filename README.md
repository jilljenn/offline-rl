# Offline RL

## Get the data

- [RoboMission 2019-12 data](https://github.com/adaptive-learning/adaptive-learning-research/tree/master/data/robomission-2019-12): extract `robomission-2019-12-10/attempts.csv` and put it in `data/` folder
- Our reformat of [Assistments](https://jiji.cat/weasel2018/data.csv) to put in `data/assistments09`.

## Run the experiments

    uv run run.py robo elo   # Generates robo-elo-new.pickle
    uv run run.py assistments-skill elo   # Generates assistments-skill-elo-new.pickle
    uv run --with jupyter jupyter notebook
    # Execute Results.ipynb (with `DATA = 'robo'`) to generate the plots PDF and PNG

- Robo should execute within seconds on a good machine. Longest cell in Jupyter notebook takes approx 10 seconds.
- Assistments skill takes 55 s on a good machine. Longest cell takes 17 seconds.
