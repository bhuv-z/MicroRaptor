# %%
from datetime import datetime
import functools
import os
import webbrowser as wb
import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# from bolt import Bolt
import brax
# from brax import envs
import custom_envs
from brax.training import ppo, sac
from brax.io import html as html

# %%
env_fn = custom_envs.create_fn(env_name="bolt")
env = env_fn()
jit_env_reset = jax.jit(env.reset)
state = jit_env_reset(rng=jax.random.PRNGKey(seed=0))

def visualize(sys, qps):
    """Renders a 3D visualization of the environment."""
    firefox = '/mnt/c/Program Files/Firefox/firefox.exe'
    wb.register('firefox', None, wb.BackgroundBrowser(firefox))
    html.save_html("test_file_1.html",sys,qps)
    # wb.get('firefox').open("test_file_1")

visualize(env.sys, [state.qp])

# %%
# We determined some reasonable hyperparameters offline and share them here. # 30000000
train_fn = functools.partial(
      ppo.train, num_timesteps = 10, log_frequency = 20,
      reward_scaling = 10, episode_length = 1000, normalize_observations = True,
      action_repeat = 1, unroll_length = 5, num_minibatches = 32,
      num_update_epochs = 4, discounting = 0.97, learning_rate = 3e-4,
      entropy_cost = 1e-2, num_envs = 2048, batch_size = 1024
  )

min_y = 10
max_y = 6000
xdata = []
ydata = []
times = [datetime.now()]


def progress(num_steps, metrics):
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'])
  plt.xlim([0, train_fn.keywords['num_timesteps']])
  plt.ylim([min_y, max_y])
  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.plot(xdata, ydata)
  plt.show()
print("training")
inference_fn, params, _ = train_fn(environment_fn=env_fn, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
rng = jax.random.PRNGKey(seed=0)
state = jit_env_reset(rng=rng)
while not state.done:
  rollout.append(state)
  act_rng, rng = jax.random.split(rng)
  act = jit_inference_fn(params, state.obs, act_rng)
  state = jit_env_step(state, act)
# print(rollout)

def visualize2(sys, qps):
    """Renders a 3D visualization of the environment."""
    firefox = '/mnt/c/Program Files/Firefox/firefox.exe'
    wb.register('firefox', None, wb.BackgroundBrowser(firefox))
    html.save_html("test_file_2.html",sys,qps)
    # wb.get('firefox').open("test_file_1")

visualize2(env.sys, [s.qp for s in rollout])


pass
# %%
