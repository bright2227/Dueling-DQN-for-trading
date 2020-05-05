import random
import numpy as np
import tensorflow as tf
import keras.backend as K
import os
from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Input, Dense, Add, Subtract, Lambda,LSTM
from keras.models import Model
from keras.optimizers import Adam
from utils.memory_buffer import MemoryBuffer


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, buffer_size, state_size, action_size=3, ):
        
        # agent config
        self.buffer = MemoryBuffer(buffer_size, True)
        
        self.state_size = state_size    	# normalized previous days
        self.action_size = action_size           		# [sit, buy, sell]
        self.inventory = []
        self.first_iter = True
        self.last_iter = False
        self.n_iter = 1

        # model config
        self.gamma = 0.95 # affinity for long term reward
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.optimizer = Adam(lr=self.learning_rate)
        # target network
         
        self.model = self._model() 
        self.target_model = clone_model(self.model)         
        self.target_model.set_weights(self.model.get_weights())


    def _model(self):
        """Creates the model
        """

        inputs = Input(shape=self.state_size)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)

        value = Dense(self.action_size, activation='linear')(x)
        a = Dense(self.action_size, activation='linear')(x)
        meam = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, meam])
        q = Add()([value, advantage])

        model = Model(inputs=inputs, outputs=q)

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= epsilon:
            return random.randrange(self.action_size)
        
        if self.last_iter:
            self.last_iter = False
            return self.action_size-1 # make a definite sell on the last iter        

        if self.first_iter:
            self.first_iter = False
            if not is_eval:
                return random.randrange(self.action_size) # make a definite buy on the first iter
            else:
                return 0
            
        state = state.reshape( (-1,)+ self.state_size )
        action_probs = self.model.predict(state)
        return np.argmax(action_probs[0])
    
    def epsilon_decay(self, epsilon, epsilon_min, epsilon_decay):
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        return epsilon
    
    def remember_sumtree(self, state, action, reward, new_state, done, ):
        """ Store experience in memory buffer
        """
        
        state = state.reshape( (-1,)+ self.state_size )
        new_state = new_state.reshape( (-1,)+ self.state_size )

        q_val = self.model.predict(state)
        q_val_t = self.target_model.predict(new_state)
        next_best_action = np.argmax(q_val)
        new_val = reward + self.gamma * q_val_t[0, next_best_action]
        td_error = abs(new_val - q_val + 1e-8 )[0]        

        self.buffer.memorize(state, action, reward, done, new_state, td_error)
        
    def target_model_update(self, done, tau=0.1, type='reset', reset_every=5000):
        
        if type == 'reset':
            if self.n_iter % reset_every == 0:
                print('update target model')
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

        if type == 'transfer':
            if done:
                W = self.model.get_weights()
                tgt_W = self.target_model.get_weights()
                for i in range(len(W)):
                    tgt_W[i] = tau * W[i] + (1 - tau) * tgt_W[i]
                self.target_model.set_weights(tgt_W)
        

    def train_experience_replay_sumtree( self, batch_size,):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        
        state, action, reward, done, new_state, idx = self.buffer.sample_batch(batch_size)
       
        state = state.reshape( (-1,)+ self.state_size )
        new_state = new_state.reshape( (-1,)+ self.state_size )
        
        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.model.predict(state)

        next_q = self.model.predict(new_state)
        q_targ = self.target_model.predict(new_state)

        for i in range(state.shape[0]):
            old_q = q[i, action[i]]
            if done[i]:
                q[i, action[i]] = reward[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, action[i]] = reward[i] + self.gamma * q_targ[i, next_best_action]
            self.buffer.update(idx[i], abs(old_q - q[i, action[i]]))

        # Train on batch
        loss = self.model.fit((state), q, epochs=1, verbose=0).history["loss"][0]   

        return loss
    
    
    def save(self,name):
        if not os.path.exists('save/'+ name):
            os.makedirs('save/'+ name)        
            np.save('save/' + name + '/data.npy', self.buffer.buffer.data)
            np.save('save/' + name + '/tree.npy', self.buffer.buffer.tree)
            self.model.save('save/' + name +'/model.h5')
        else:
            print('already exist, please check.')


    def load(self,name):
        if not os.path.exists('save/'+ name):
            print('not exist, please check.')   
        else:
            self.buffer.buffer.data = np.load('save/' + name + '/data.npy',allow_pickle=True)
            self.buffer.buffer.tree = np.load('save/' + name + '/tree.npy',allow_pickle=True)
            self.model=load_model('save/' + name +'/model.h5')


