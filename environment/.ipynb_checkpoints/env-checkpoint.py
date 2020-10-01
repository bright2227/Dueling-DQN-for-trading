import numpy as np
import pandas as pd


class Environment:

    def __init__(self, data, ):
        self.data = data         
        self.inventory = []
        self.hold = 0
        self.hold_day=0
        self.t = 0
        self.data_length = data.index.max()
        self.state = self.get_state()
        self.history = [0]
        
    def get_state(self,):
        res = self.data.loc[self.t].tolist()[1:-2] #avoid open,close and date
        
        if self.hold == 1:
            self.hold_day += 1
            delta_reserved = (self.data['Adj Close'].loc[self.t])/self.inventory[0]
            reward_reserved = np.log(delta_reserved-0.007)
        else:
            reward_reserved = 0
            self.hold_day = 0
        
        res.extend([self.hold, reward_reserved/(self.hold_day+1), self.hold_day*0.005])

        return np.array([res])
    
    def reset(self,):
        self.inventory = []
        self.hold=0
        self.hold_day=0
        self.t = 0
        self.state = self.get_state()
        self.history = [0]
        
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
        if (act == 0) and len(self.inventory) == 0:
            self.inventory.append(self.data['Adj Open'].loc[self.t])
            self.hold = 1        
        
        # SELL
        elif (act == 2) and len(self.inventory) > 0:
            bought_price = self.inventory.pop(0)
            delta = (self.data['Adj Open'].loc[self.t])/bought_price
            reward = np.log(delta-0.007)
            self.hold = 0
            
        self.history.append(self.hold)
        self.t = self.t + 1
        self.state = self.get_state()
        
        if self.t == self.data_length:
            done = True       
        
        return self.state, reward, done,


