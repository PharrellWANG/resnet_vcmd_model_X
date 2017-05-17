import math

max_learning_rate = 0.3
min_learning_rate = 0.0001
decay_speed = 15000

for train_step in range(10000):
    # train_step += 1
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-train_step / decay_speed)
    print(learning_rate)
