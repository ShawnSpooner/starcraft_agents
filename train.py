import sys
import numpy as np

from starcraft_agents.a2c_agent import A2CAgent
from pysc2.env.sc2_env import SC2Env
from pysc2.env.environment import TimeStep, StepType
from absl import flags
from absl.flags import FLAGS
from multiprocessing import Process, Pipe
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, CloudpickleWrapper
from functools import partial
import visdom
from torchnet.logger import VisdomPlotLogger, VisdomLogger

from starcraft_agents.common import SC2TorchEnv, SC2TorchEnv, SC2ProcVec


def train():
    env_args = dict(
        map_name=FLAGS.map_name,
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        screen_size_px=(FLAGS.resolution,) * 2,
        minimap_size_px=(FLAGS.resolution,) * 2,
        visualize=FLAGS.visualize
    )
    max_frames = FLAGS.frames * 1e6
    total_frames = 0
    vis = visdom.Visdom()
    vis.close(env=FLAGS.expirement_name, win=None)

    envs = SC2ProcVec([partial(SC2TorchEnv, env_args) for i in range(FLAGS.n_envs)])
    print(f"Starting {FLAGS.n_envs} workers")

    try:
            agent = A2CAgent(screen_width=FLAGS.resolution,
                             screen_height=FLAGS.resolution,
                             expirement_name=FLAGS.expirement_name,
                             learning_rate=FLAGS.learning_rate,
                             num_processes=FLAGS.n_envs,
                             value_coef=FLAGS.value_weight,
                             entropy_coef=FLAGS.entropy_weight,
                             continue_training=FLAGS.continue_training,
                             horizon=FLAGS.horizon)

            num_processes = FLAGS.n_envs
            horizon = FLAGS.horizon

            timesteps = envs.reset()
            agent.reset()

            while total_frames * num_processes <= max_frames:
                total_frames += 1
                step = total_frames % horizon
                agent.finish_step()
                actions = [agent.step(step, p, TimeStep(*t))
                           for p, t in enumerate(timesteps.reshape(num_processes, 4))]

                if step == 0:
                    agent.rollout()
                    agent.reset()

                timesteps = envs.step(actions)

    except KeyboardInterrupt:
        pass
    finally:
        envs.close()

    print(f"Training done after {total_frames} steps")

def main():
    flags.DEFINE_string("map_name", "FindAndDefeatZerglings", "Which map to use")
    flags.DEFINE_boolean("continue_training", False, "Continue with training?")
    flags.DEFINE_integer("frames", 10, "Number of frames in millions")
    flags.DEFINE_integer("horizon", 40, "Number of steps before cutting the trajectory")
    flags.DEFINE_integer("step_mul", 8, "sc2 frame step size")
    flags.DEFINE_integer("n_envs", 1, "Number of sc2 environments to run in parallel")
    flags.DEFINE_integer("resolution", 32, "sc2 resolution")
    flags.DEFINE_float("learning_rate", 7e-4, "learning rate")
    flags.DEFINE_boolean("visualize", False, "show pygame visualisation")
    flags.DEFINE_float("value_weight", 0.5, "value function loss weight")
    flags.DEFINE_float("entropy_weight", 0.01, "entropy loss weight")
    flags.DEFINE_string("expirement_name", "lings_1", "What shall we call this model?")

    FLAGS(sys.argv)

    train()


if __name__ == '__main__':
    main()
