import sys

from starcraft_agents.a2c_agent import A2CAgent
from pysc2.env.sc2_env import SC2Env
from absl import flags
from absl.flags import FLAGS


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
    try:
        with SC2Env(**env_args) as env:
            agent = A2CAgent(screen_width=FLAGS.resolution,
                             screen_height=FLAGS.resolution,
                             expirement_name=FLAGS.expirement_name,
                             learning_rate=FLAGS.learning_rate)

            while total_frames <= max_frames:
                timesteps = env.reset()
                agent.reset()

                while True:
                    total_frames += 1
                    actions = [agent.step(timestep) for timestep in timesteps]
                    if max_frames and total_frames >= max_frames:
                        break
                    if timesteps[0].last():
                        break

                    timesteps = env.step(actions)

    except KeyboardInterrupt:
        pass

    print(f"Training done after {total_frames} steps")
    env.close()


def main():
    flags.DEFINE_string("map_name", "FindAndDefeatZerglings", "Which map to use")
    flags.DEFINE_integer("frames", 10, "Number of frames in millions")
    flags.DEFINE_integer("step_mul", 8, "sc2 frame step size")
    flags.DEFINE_integer("n_envs", 1, "Number of sc2 environments to run in parallel")
    flags.DEFINE_integer("resolution", 32, "sc2 resolution")
    flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
    flags.DEFINE_boolean("visualize", False, "show pygame visualisation")
    flags.DEFINE_float("value_weight", 1.0, "value function loss weight")
    flags.DEFINE_float("entropy_weight", 1e-5, "entropy loss weight")
    flags.DEFINE_string("expirement_name", "lings_1", "What shall we call this model?")

    FLAGS(sys.argv)
    train()


if __name__ == '__main__':
    main()
