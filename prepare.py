import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import QuantileTransformer
import logging


MINIMAX = ['max_context', 'min_context', 'prev_outcome']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def irt_proba(theta, diff):
    return sigmoid(theta - diff)


def get_data(DATA):
    if DATA == 'robo':
        df = pd.read_csv('data/attempts.csv')
        df['outcome'] = df['solved'].astype(int)
        df['user'] = np.unique(df['student'], return_inverse=True)[1]
        df['unordered_item'] = np.unique(df['problem'], return_inverse=True)[1]
    else:
        df = pd.read_csv('data/assistments09/data.csv')
        if DATA == 'assistments-small':
            top_items = df['item'].value_counts().head(1000).index.tolist()
            df = df.query("item in @top_items")
        elif DATA == 'assistments-skill':  # Assistments skill
            df['item'] = df['skill']
        df['user'] = np.unique(df['user'], return_inverse=True)[1]
        df['unordered_item'] = np.unique(df['item'], return_inverse=True)[1]
        df['outcome'] = df['correct']
        df['start'] = df.index
    df = df.sort_values(['user', 'start'])
    return df


def populate_data(df, DATA):
    # df = pd.read_csv('data/dummy/data.csv')
    # df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    model = Pipeline([('onehot', OneHotEncoder()),
                      ('lr', LogisticRegression(solver='liblinear', C=1e10, random_state=42))])

    model.fit(df[['user', 'unordered_item']], df['outcome'])  # Train
    # print(model.predict_proba(df_test[['student', 'problem']]))  # Test
    # Assist 9 s
    # RoboMission 19 s

    n_items = df['unordered_item'].nunique()

    difficulties = -model['lr'].coef_[:, -n_items:].flatten()
    # difficulties = samuel_df['diffik']
    df['diff'] = df['unordered_item'].map(lambda x: difficulties[x])

    # unranked_diff = df_train[['unordered_item', 'diff']].drop_duplicates().sort_values('unordered_item')['diff'].values
    # sorted_order = unranked_diff.argsort()

    #df_train = df.query("in_train == True")
    #df_test = df.query("in_train == False")

    ranked_diff = df[['unordered_item', 'diff']].drop_duplicates().sort_values('diff')
    inverse_dict = dict(zip(ranked_diff['unordered_item'], range(len(ranked_diff))))

    df['item'] = df['unordered_item'].map(lambda x: inverse_dict[x])
    df['action'] = df['item']
    df['position'] = 0

    if DATA == 'assistments':        
        binner = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')  # was: uniform
        df['bin_diff'] = binner.fit_transform(df[['diff']])
        # df['bin_diff'].hist(bins=50)

        df['action'] = df['bin_diff'].astype(int)

        df['mean_diff'] = df.groupby(["action"])['diff'].transform('mean')
        df['diff'] = df['mean_diff']
        difficulties = np.array(sorted(df['diff'].unique()))  # Overwrite difficulties

    # df['action'].value_counts()
    n_actions = df['action'].nunique()

    logging.warning('Number of unique items: %d', df['item'].nunique())
    logging.warning('Number of unique actions: %d', df['action'].nunique())
    logging.warning('Number of unique difficulties: %d', df['diff'].nunique())

    df['reward_if_okay'] = df['diff'] - difficulties.min()
    df['reward'] = df['outcome'] * (df['diff'] - difficulties.min())
    return n_actions, difficulties


def compute_contexts(df, difficulties, CONTEXT):
    if CONTEXT == 'elo':
        K = 0.5
        contexts = []
        last_user = None
        context = 0
        for _, row, in df.iterrows():
            if last_user is None or row['user'] != last_user:
                context = 0
                last_user = row['user']
            contexts.append(context)
            proba = irt_proba(context, row['diff'])
            context += K * (row['outcome'] - proba)
        df['context'] = contexts
        # Robo: 21 s, 8 s if performant
        # Assist: 8 s perf

        df['elo_context'] = df['context']
        qt = QuantileTransformer()
        df['qt_context'] = qt.fit_transform(df[['elo_context']])
        df['full_context'] = df['elo_context']
        df['context'] = df['full_context'].round(1)
        # df['context'].hist(bins=101)
        df['min_context'] = df['elo_context']
        df['max_context'] = df['elo_context']

    elif CONTEXT == 'minimax':
        # Compute minimax contexts
        min_contexts = []
        max_contexts = []
        prev_outcomes = []
        last_user = None
        inf_rew = (difficulties - difficulties.min()).max()
        min_context = inf_rew
        max_context = prev_outcome = 0
        for _, row, in df.iterrows():
            if last_user is None or row['user'] != last_user:
                min_context = inf_rew
                max_context = prev_outcome = 0
                last_user = row['user']
            min_contexts.append(min_context)
            max_contexts.append(max_context)
            prev_outcomes.append(prev_outcome)
            # proba = irt_proba(context, row['diff'])
            if row['outcome'] == 0:  # Failed
                this_reward = (row['diff'] - difficulties.min())
                min_context = min(min_context, this_reward)
            else:
                max_context = max(max_context, row['reward'])
            prev_outcome = row['outcome']
        df['min_context'] = min_contexts
        df['max_context'] = max_contexts
        df['prev_outcome'] = prev_outcomes
        # Robo: 9 s if performant
        # Assist: 7 s perf

    df['mid_context'] = (df['min_context'] + df['max_context']) / 2

    if CONTEXT == 'elo':
        df['context_id'] = df['context']
    else:
        df['minimax_context'] = df['min_context'].astype(str) + df['max_context'].astype(str) + df['prev_outcome'].astype(str)
        sorted_minimax = df[MINIMAX + ['minimax_context']].drop_duplicates().sort_values(by=MINIMAX).reset_index()
        minimax_order = dict(zip(sorted_minimax['minimax_context'], sorted_minimax.index))
        df['context_id'] = df['minimax_context'].map(minimax_order)



def compute_pscore(df, CONTEXT):
    if CONTEXT == 'elo':
        # Computing pscore Elo
        df['context_action_occ'] = df.groupby(["context", "action"])['reward'].transform('count')
        df['context_occ'] = df.groupby(["context"])['reward'].transform('count')
        df['pscore'] = df['context_action_occ'] / df['context_occ']
        # df.head()
        assert all(abs(df[['context', 'action', 'pscore']].drop_duplicates().groupby('context')['pscore'].sum().values - 1) < 1e-8)

    else:
        # Computing pscore minimax
        df['context_action_occ'] = df.groupby(MINIMAX + ["action"])['reward'].transform('count')
        df['context_occ'] = df.groupby(MINIMAX)['reward'].transform('count')
        df['pscore'] = df['context_action_occ'] / df['context_occ']
        # df.head()

    df['reward_on_pscore'] = df['reward'] / df['pscore']

