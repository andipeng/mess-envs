from stable_baselines3 import PPO

import metaworld
import random
ml1 = metaworld.ML1('stick-push-v2') # Construct the benchmark, sampling tasks
env = ml1.train_classes['stick-push-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)

model = PPO("MlpPolicy", env, verbose=5)
model.learn(total_timesteps=1000000)


 
