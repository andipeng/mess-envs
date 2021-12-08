import random
import click

from stable_baselines3 import SAC
import metaworld
import tensorboard

DESC = '''
Script to train singletask (ML1 or MT1) envs in metaworld.\n
USAGE:\n
    $ python train_singletask.py --env_name button-press-v2 --timesteps 10000000 \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment from ML1 or MT1', default='button-press-v2')
@click.option('--seed', type=int, help='timesteps to train', default=0)
@click.option('--algo', type=str, help='algo from sb3', default='sac')
@click.option('--logdir', type=str, help='dir to save logs', default='./logs/')
@click.option('--timesteps', type=int, help='timesteps to train', default=10000000)

def main(env_name, seed, algo, logdir, timesteps):
    # constructs the benchmark
    # print(metaworld.ML1.ENV_NAMES)
    ml1 = metaworld.ML1(env_name, seed=seed)
    env = ml1.train_classes[env_name]()
    task = random.choice(ml1.train_tasks) # randomly samples 1 task aka goal
    env.set_task(task)

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir+env_name+'/')
    model.learn(total_timesteps=timesteps)

if __name__ == '__main__':
    main()