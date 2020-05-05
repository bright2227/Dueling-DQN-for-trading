import os
import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm_notebook


def get_state(data, t, state_sup):
    """Returns an n-day state representation ending at time t
    """
    res = data.loc[t].tolist()[1:-2] #avoid open,close and date
    res.extend(state_sup)
    
    return np.array([res])

def evaluate_model(agent, data, history ):
    total_profit = 0
    data_length = data.index.max()
    epsilon = 0
    hold_day = 0 
    inventory = []
    hold=0
    reward_reserved = 0
    
    state = get_state(data, 0, [hold, reward_reserved/(hold_day+1), hold_day*0.005])
    agent.first_iter = True
    agent.last_iter = False

    for t in range(1, data_length):        
        reward = 0
                
        # select an action
        if (t == data_length - 1):            
            agent.last_iter = True   
            
        action = agent.act(state, epsilon, is_eval=True)

        # BUY
        if (action == 0) and len(inventory) == 0:
            inventory.append(data['Adj Open'].loc[t])
            hold = 1
            history.append((1,t,reward))        
        # SELL
        elif action == 2 and len(inventory) > 0:
            bought_price = inventory.pop(0)
            delta = (data['Adj Open'].loc[t])/bought_price
            reward = np.log(delta-0.007)
            total_profit += reward
            hold = 0
            history.append((0,t,reward))            
        
        if hold == 1:
            hold_day += 1
            delta_reserved = (data['Adj Close'].loc[t])/inventory[0]
            reward_reserved = np.log(delta_reserved-0.007)
        else:
            reward_reserved = 0
            hold_day=0
            
        done = (t == data_length - 1)
        next_state = get_state(data, t , [hold, reward_reserved/(hold_day+1), hold_day*0.005])
        state = next_state
        if done:
            return total_profit, history

def plot(data,history):
    result = data[['Adj Close','date']]
    result['sig']=np.nan
    
    for log in history :
        result['sig'].iloc[log[1]]=log[0]
        
    result = result.fillna(method='ffill')

    result.set_index('date')['Adj Close'].plot()
    result.set_index('date')['sig'].plot(secondary_y=True)
    return result