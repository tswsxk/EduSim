# coding: utf-8
# 2020/4/29 @ tongshiwei

from tensorboardX import SummaryWriter
import logging
from longling import json_load
from longling import path_append, abs_current_dir
from EduSim.Envs.meta import Env
from .Learner import Learner
from .Exerciser import Exerciser
from copy import deepcopy
from EduSim.utils.io_lib import load_ks_from_csv
from EduSim.utils import board_episode_callback
from EduSim.SimOS import train_eval

__all__ = ["TMSEnv", "tms_train_eval"]

ROOT = path_append(abs_current_dir(__file__), "meta_data")

ENV_META = {
    "binary": path_append(ROOT, "binary", to_str=True),
    "tree": path_append(ROOT, "tree", to_str=True),
}


def tms_train_eval(agent, env: Env, max_steps: int = None, max_episode_num: int = None, n_step=False,
                   train=False,
                   logger=logging, level="episode", board_dir=None):
    def summary_callback(rewards, infos, logger):
        expected_reward = sum(rewards) / len(rewards)

        logger.info("Expected Reward: %s" % expected_reward)

        return expected_reward, infos

    sw = None
    if board_dir is not None:
        sw = SummaryWriter(board_dir)

        def episode_callback(episode, reward, *args):
            return board_episode_callback(episode, reward, sw)

    else:
        episode_callback = None

    train_eval(
        agent, env,
        max_steps, max_episode_num, n_step, train,
        logger, level,
        episode_callback=episode_callback,
        summary_callback=summary_callback,
    )

    if board_dir:
        sw.close()


def load_matrix(filename):
    from collections import defaultdict
    meta_data = json_load(filename)

    for i in range(len(meta_data)):
        if isinstance(meta_data[i], dict):
            converted_dict = {}
            for k, v_dict in meta_data[i].items():
                converted_dict[int(k)] = defaultdict(float)
                for key, value in v_dict.items():
                    converted_dict[int(k)][int(key)] = value
            meta_data[i] = converted_dict

    return meta_data


NO_MEASUREMENT_ERROR = 0
MEASUREMENT_ERROR = 1

NAME = [
    "binary",
    "tree",
]

MODE = {
    "nme": NO_MEASUREMENT_ERROR,
    "me": MEASUREMENT_ERROR,
}


class TMSEnv(Env):
    def __init__(self, name, measure_item_for_each_skill=2, mode="me"):
        """

        Parameters
        ----------
        name:
            binary or tree
        measure_item_for_each_skill
        mode
        """
        super(TMSEnv, self).__init__()

        self.ks = load_ks_from_csv(ENV_META[name] + "_ks.csv")
        self._learner = Learner(
            load_matrix(ENV_META[name] + ".json"),
            json_load(ENV_META[name] + "_state.json"),
            json_load(ENV_META[name] + "_init_state.json")
        )
        self._meta = json_load(ENV_META[name] + "_meta.json")
        self._skill_num = self._meta["skill_num"]
        self._begin_state = None
        self.mode = MODE[mode]
        self._measure_item_for_each_skill = measure_item_for_each_skill
        self._exerciser = Exerciser(self._skill_num, measure_item_for_each_skill)

    @property
    def description(self) -> dict:
        return {
            "ks": self.ks,
            "action_space": list(range(self.action_num)),
        }

    @property
    def exercise_bank(self):
        return self._exerciser.exercise_bank

    @property
    def action_num(self):
        return self._skill_num

    def render(self, mode='human'):
        if mode == "log":
            return "state: %s" % str(self._learner.state)

    def reset(self):
        self._begin_state = None

    def step(self, learning_item_id, *args, **kwargs):
        proficiency = sum(self._learner.state)
        self._learner.learn(learning_item_id)
        reward = sum(self._learner.state) - proficiency
        if self.mode == NO_MEASUREMENT_ERROR:
            observation = self._learner.state
        else:
            observation = self._exerciser.exam(self._learner, *range(self._skill_num))
        return observation, reward, len(self._learner.state) == sum(
            self._learner.state), None

    def n_step(self, learning_path, *args, **kwargs):
        return zip(*[self.step(learning_item) for learning_item in learning_path])

    def begin_episode(self, *args, **kwargs):
        self._learner.reset()
        self._begin_state = deepcopy(self._learner.state)
        return self._learner.profile

    def end_episode(self, *args, **kwargs):
        if self.mode == NO_MEASUREMENT_ERROR:
            observation = self._learner.state
        else:
            observation = self._exerciser.exam(self._learner, *range(self._skill_num))
        reward = sum(self._learner.state) - sum(self._begin_state)
        return observation, reward, len(self._learner.state) == sum(self._learner.state), None
