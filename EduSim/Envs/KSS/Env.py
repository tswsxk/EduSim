# coding: utf-8
# 2019/11/26 @ tongshiwei

import random
import networkx as nx
from tqdm import tqdm
import json
from longling import wf_open

from EduSim.Envs.Env import Env
from EduSim.Envs.Reward import get_reward

from .KS import get_knowledge_structure
from .Learner import Learner
from .QBank import QBank
from .Tester import Tester

ORDER_RATIO = 1


class KSS(Env):
    def __init__(self, learner_num=4000, seed=10, logger=None, log_rf=None, **kwargs):
        super(KSS, self).__init__(logger=logger, log_rf=log_rf, **kwargs)

        random.seed(seed)

        # initialize knowledge structure
        self.ks = get_knowledge_structure()

        # initialize exercise attributes
        self.topo_order = list(nx.topological_sort(self.ks))
        self.default_order = [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]
        self.q_bank = QBank(self.ks.number_of_nodes(), self.default_order)
        self.review_times = 1

        # initialize tester
        self.tester = Tester(self.q_bank)

        # initialize learners
        self.learners = self.generate_learners(learner_num)

        # controller variables
        random.seed(None)
        self.reward = get_reward()

    def remove_invalid_sample(self, idx):
        del self.learners[idx]

    def begin_episode(self) -> Learner:
        while True:
            try:
                _idx = random.randint(0, len(self.learners) - 1)
            except ValueError:
                raise ValueError

            self._learner = self.learners[_idx]
            self.tester.begin_episode(self._learner)
            if self.is_valid_sample(self._learner):
                self._path = []
                return self._learner
            else:  # pragma: no cover
                self.tester.end_episode()
                self.remove_invalid_sample(_idx)

    def summary_episode(self, *args, **kwargs) -> dict:
        _learner = self._learner
        _path = self._path

        tester_summary = self.tester.end_episode(self._path)

        summary = {
            "learner_id": _learner.id,
            "path": _path,
            "reward": self.episode_reward(
                tester_summary["initial_score"],
                tester_summary["final_score"],
                tester_summary["full_score"],
            ),
            "evaluation": tester_summary["evaluation"]
        }

        return summary

    def test(self, exercise):
        result = {
            "exercise": exercise,
            "correct": self.tester.test(exercise)
        }
        self._learner.exercise_history.append([result["exercise"], result["correct"]])
        return result

    ###########################################################
    #                   individual functions                  #
    ###########################################################

    # #################### learners ###########################
    def generate_learners(self, learner_num, step=20):
        initial_learner_abilities = self._get_learner_ability(learner_num)
        learners = []

        for learner_ability in tqdm(initial_learner_abilities, "loading data"):
            learner = Learner(
                initial_state=learner_ability,
                knowledge_structure=self.ks,
                learning_target=set(random.sample(self.ks.nodes, random.randint(3, len(self.ks.nodes)))),
            )
            self._learner_warm_up(learner, step)
            learners.append(learner)

        return learners

    @staticmethod
    def _get_learner_ability(learner_num):
        return [[random.randint(-3, 0) - (0.1 * i) for i in range(10)] for _ in range(learner_num)]

    # #################### capacity growth #########################
    def _learner_warm_up(self, learner: Learner, step):  # pragma: no cover
        self.tester.begin_episode(learner)
        self._learner = learner
        if random.random() < ORDER_RATIO:
            while len(learner.exercise_history) < step:
                if learner.exercise_history and learner.exercise_history[-1][1] == 1 and len(
                        set([e[0] for e in learner.exercise_history[-3:]])) > 1:
                    for _ in range(self.review_times):
                        if len(learner.exercise_history) < step - self.review_times:
                            learning_item = exercise = learner.exercise_history[-1][0]
                            learner.learn(learning_item)
                            self.test(exercise)
                        else:
                            break
                    learning_item = learner.learning_history[-1]
                elif learner.exercise_history and learner.exercise_history[-1][1] == 0 and random.random() < 0.7:
                    learning_item = learner.exercise_history[-1][0]
                elif random.random() < 0.9:
                    for learning_item in self.topo_order:
                        if self.tester.test(learning_item, binary=False) < 0.6:
                            break
                    else:
                        break
                else:
                    learning_item = random.randint(0, len(self.topo_order) - 1)

                learner.learn(learning_item)
                self.test(learning_item)
        else:
            while len(learner.learning_history) < step:
                if random.random() < 0.9:
                    for learning_item in self.default_order:
                        if self.tester.test(learning_item, binary=False) < 0.6:
                            break
                    else:
                        break
                else:
                    learning_item = random.randint(0, len(self.topo_order) - 1)
                learner.learn(learning_item)
                self.test(learning_item)
        self.tester.end_episode()
        self._learner = None
        assert len(learner.exercise_history) <= step, len(learner.exercise_history)

    # ###################### off-line data ############################
    def dump_id2idx(self, filename):
        self.ks.dump_id2idx(filename)

    def dump_graph_edges(self, filename):
        self.ks.dump_graph_edges(filename)

    def dump_kt(self, learner_num, filename, step=50):
        learners = self.generate_learners(learner_num)

        with wf_open(filename) as wf:
            for learner in tqdm(learners, "kss for kt"):
                self._learner_warm_up(learner, step)
                print(json.dumps(learner.exercise_history), file=wf)
