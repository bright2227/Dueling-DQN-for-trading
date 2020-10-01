import os
import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm_notebook


def training_evaluate(agent, env, ):
    
    train_profit, done = 0, False
    epsilon = 0
    
    state = env.reset()

    while not done:

        # select an action
        action = agent.act(state, epsilon,)
        next_state, reward, done, = env.step(action)
        train_profit += reward      
        
        state = next_state
        
        if done:
            return train_profit, env.history
        

def plot(data, history, episode):
    result = data[['Adj Close','date']]
    result['hold'] = pd.Series(history[episode], data.index)
    
    result.set_index('date')['Adj Close'].plot()
    result.set_index('date')['hold'].plot(secondary_y=True)
    return result