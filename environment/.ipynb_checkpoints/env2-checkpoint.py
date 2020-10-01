import numpy as np
import pandas as pd


class Environment:

    def __init__(self, data, capital=1000):
        self.data = data         
        self.data_length = data.index.max()

        self.t = 0
        self.history = []
        self.inventory = []
        self.capital_initial = capital
        self.capital = capital
        self.capital_ratio = 0.5
        self.hold_ratio = 0
        self.state = self.get_state()

    def get_state(self,):
        res = self.data.loc[self.t].tolist()[1:-2] #avoid open,close and date
        
        if self.hold_ratio > 0:
            bought_price = sum(self.inventory)/len(self.inventory)
            delta = (self.data['Adj Close'].loc[self.t])/bought_price
            reward_reserved = (delta-1.007)*self.hold_ratio
        else:
            reward_reserved = 0
        
        res.extend([self.capital_ratio, self.hold_ratio, reward_reserved])

        return np.array([res])
    
    def reset(self, capital=1000):
        self.t = 0
        self.history = []
        self.inventory = []
        self.capital_initial = capital
        self.capital = capital
        self.capital_ratio = 0.5
        self.hold_ratio = 0
        self.state = self.get_state()
        
        return self.state
        
    
    def step(self, act):
        
        reward = 0
        done = False
        
        #Force it buy stock at first, sell at the end
        if self.t == 0:
            act = 0
        elif self.t == self.data_length-1:
            act = 2        
        else:
            pass
                
        # BUY
        if (act == 0) and (self.capital > self.data['Adj Open'].loc[self.t]):
            self.inventory.append(self.data['Adj Open'].loc[self.t])
            self.capital = self.capital - self.data['Adj Open'].loc[self.t]
            self.hold_ratio = self.hold_ratio + self.capital_ratio - self.capital/self.capital_initial*0.5
            self.capital_ratio =  self.capital/self.capital_initial*0.5
                
        
        # SELL
        elif (act == 2) and len(self.inventory) > 0:
            bought_price = sum(self.inventory)/len(self.inventory)
            delta = (self.data['Adj Open'].loc[self.t])/bought_price
            reward = (delta-1.007)*self.hold_ratio
            self.capital_ratio =  (delta-0.007)*self.hold_ratio + self.capital_ratio
            self.capital = self.capital_ratio * self.capital_initial / 0.5
            self.inventory = []
            self.hold_ratio = 0
            
        self.history.append(act)
        self.t = self.t + 1
        self.state = self.get_state()
        
        if (self.t == self.data_length) or (self.capital_ratio < 0):
            done = True       
        
        return self.state, reward, done,


