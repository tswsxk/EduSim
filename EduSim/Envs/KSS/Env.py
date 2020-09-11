# coding: utf-8
# 2020/4/30 @ tongshiwei

import networkx as nx
import random
from EduSim.Envs.meta import Env

from .Learner import LearnerGroup
from .KS import get_knowledge_structure
from .QBank import QBank
from .Exerciser import Exerciser
from EduSim.Envs.shared.KSS_KES import episode_reward, kss_kes_train_eval as kss_train_eval

__all__ = ["KSSEnv", "kss_train_eval"]

RANDOM = 0
LOOP = 1
INF = 2

MODE = {
    "random": RANDOM,
    "loop": LOOP,
    "inf": INF,
}


class KSSEnv(Env):
    def __init__(self, learner_num=4000, seed=10, order_ratio=1, initial_step=20, mode="inf"):
        random.seed(seed)

        self.mode = MODE[mode]

        self.ks = get_knowledge_structure()
        self.default_order = [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]
        self.q_bank = QBank(self.ks.number_of_nodes(), self.default_order)
        self._exerciser = Exerciser(self.q_bank)

        self._learner_num = learner_num
        self.learner_group = LearnerGroup(self.ks)
        self._order_ratio = order_ratio
        self._topo_order = list(nx.topological_sort(self.ks))
        self._initial_step = initial_step
        self._review_times = 1

        self._idx = 0
        if self.mode in {RANDOM, LOOP}:
            self.initial_learner_group()

        self._initial_learner_state = None
        self._learner = None

    @property
    def description(self) -> dict:
        return {
            "ks": self.ks,
            "action_space": list(range(self.action_num)),
        }

    @property
    def action_num(self):
        return self.ks.number_of_nodes()

    def _initial_learner(self, learner):
        exercise_history = []

        if random.random() < self._order_ratio:
            while len(exercise_history) < self._initial_step:
                if exercise_history and exercise_history[-1][1] == 1 and len(
                        set([e[0] for e in exercise_history[-3:]])) > 1:
                    for _ in range(self._review_times):
                        if len(exercise_history) < self._initial_step - self._review_times:
                            learning_item = exercise = exercise_history[-1][0]
                            learner.learn(learning_item)
                            exercise_history.append(self._exerciser.test(exercise, learner, binary_mode=True))
                        else:
                            break
                    learning_item = exercise_history[-1][0]
                elif exercise_history and exercise_history[-1][1] == 0 and random.random() < 0.7:
                    learning_item = exercise_history[-1][0]
                elif random.random() < 0.9:
                    for learning_item in self._topo_order:
                        if self._exerciser.test(learning_item, learner, binary_mode=False)[1] < 0.6:
                            break
                    else:  # pragma: no cover
                        break
                else:
                    learning_item = random.randint(0, len(self._topo_order) - 1)

                learner.learn(learning_item)
                exercise_history.append(self._exerciser.test(learning_item, learner, binary_mode=True))
        else:
            while len(learner.learning_history) < self._initial_step:
                if random.random() < 0.9:
                    for learning_item in self.default_order:
                        if self._exerciser.test(learning_item, learner, binary_mode=False)[1] < 0.6:
                            break
                    else:
                        break
                else:
                    learning_item = random.randint(0, len(self._topo_order) - 1)
                learner.learn(learning_item)
                exercise_history.append(self._exerciser.test(learning_item, learner, binary_mode=True))
        assert len(exercise_history) <= self._initial_step, len(learner.exercise_history)
        return exercise_history

    def initial_learner_group(self):
        while len(self.learner_group) < self._learner_num:
            learner = self.learner_group.new_learner()
            exercise_history = self._initial_learner(learner)
            if sum([v for _, v in self._exerciser.exam(learner, *learner.target)]) < len(learner.target):
                learner.set_profile(exercise_history)
                self.learner_group.add(learner)

    def begin_episode(self, *args, **kwargs):
        if self.mode == RANDOM:
            learner = self.learner_group.sample()
        elif self.mode == LOOP:
            learner = self.learner_group[self._idx]
            self._idx = (self._idx + 1) % len(self.learner_group)
        elif self.mode == INF:
            while True:
                learner = self.learner_group.new_learner()
                exercise_history = self._initial_learner(learner)
                if sum([v for _, v in self._exerciser.exam(learner, *learner.target)]) < len(learner.target):
                    learner.set_profile(exercise_history)
                    break
        else:  # pragma: no cover
            raise TypeError("unknown mode %s" % self.mode)

        self._learner = learner
        self._initial_learner_state = self._learner.state
        return learner.profile

    def end_episode(self, *args, **kwargs):
        observation = self._exerciser.exam(self._learner, *self._learner.target)
        initial_score = sum([v for _, v in self._exerciser.exam(self._initial_learner_state, *self._learner.target)])
        final_score = sum([v for _, v in observation])
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}

        assert reward >= 0, "%s" % self._idx

        return observation, reward, done, info

    def step(self, learning_item_id, *args, **kwargs):
        a = sum([v for _, v in self._exerciser.exam(self._learner, *self._learner.target)])
        self._learner.learn(learning_item_id)
        observation = self._exerciser.test(learning_item_id, self._learner)
        b = sum([v for _, v in self._exerciser.exam(self._learner, *self._learner.target)])

        return observation, b - a, b == len(self._learner.target), None

    def n_step(self, learning_path, *args, **kwargs):
        exercise_history = []
        a = sum([v for _, v in self._exerciser.exam(self._learner, *self._learner.target)])
        for learning_item_id in learning_path:
            self._learner.learn(learning_item_id)
            exercise_history.append(self._exerciser.test(learning_item_id, self._learner))
        b = sum([v for _, v in self._exerciser.exam(self._learner, *self._learner.target)])
        return exercise_history, b - a, b == len(self._learner.target), None

    def reset(self):
        self._learner = None
        self._initial_learner_state = None
        self._idx = 0

    def render(self, mode='human'):
        if mode == "log":
            return "target: %s, state: %s" % (
                self._learner.target, dict(self._exerciser.exam(self._learner, *self._learner.target)))
