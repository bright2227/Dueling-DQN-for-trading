# Dueling-DQN-for-trading

important module

keras(2.3.1), tensorflow(1.14.0), numpy(1.16.5), pandas(0.24.2),tqdm


Rule

network output Buy, Sell, Hold(do nothing)

Assume all money can be divide by stock price. 

If model acts Buy, exchanges all the money into stocks, opposite to 'Sell'.



Cautions

500 eposides to reach acceptable performance on train set (TW0050 2018)by evaluate_model. If you lower the buffer memory to half (20000), it takes 1500 eposides to reach acceptable performance.

test set (TW0050 2019) inflates through most of year, can't trust the performance on it

Unsolved problems

1.saving and loading model fails

2.ploting costs too much time
