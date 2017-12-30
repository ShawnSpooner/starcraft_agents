import numpy as np

from pysc2.env.sc2_env import SC2Env
from multiprocessing import Process, Pipe
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, CloudpickleWrapper


# borrowed this code from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
def sc2_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            timestep = env.step(data)[0]
            remote.send(timestep)
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        else:
            raise NotImplementedError


class SC2ProcVec(SubprocVecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=sc2_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        obs = np.array([remote.recv() for remote in self.remotes])
        return np.stack(obs)

    def reset_worker(self, i):
        self.remotes[i].send(('reset', None))
        obs = self.remotes[i].recv()
        return np.stack(obs)


class SC2TorchEnv(SC2Env):
    def __init__(self, env_args):
        super(SC2TorchEnv, self).__init__(**env_args)
        self.observation_space = []
        self.action_space = []

    def step(self, act):
        return super(SC2TorchEnv, self).step([act])
