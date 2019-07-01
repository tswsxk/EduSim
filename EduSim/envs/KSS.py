# coding: utf-8
# create by tongshiwei on 2019/6/25
import json
import math
import random

import networkx as nx
from longling import wf_open
from tqdm import tqdm

from EduSim.envs.Env import Env, influence_control, get_reward

graph_edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (2, 8),
    (3, 4),
    (4, 8),
    (5, 4),
    (5, 9),
    (6, 7),
    (7, 8),
    (8, 9),
]

ORDER_RATIO = 1


def irt(ability, difficulty, c=0.25):
    discrimination = 5
    return c + (1 - c) / (1 + math.exp(-1.7 * discrimination * (ability - difficulty)))


class KSS(Env):
    def __init__(self, reward=get_reward(), student_num=4000, seed=10, **kwargs):
        super(KSS, self).__init__(**kwargs)
        self.path = None

        random.seed(seed)

        self.reward = reward
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(graph_edges)
        self.topo_order = list(nx.topological_sort(self.graph))
        self.default_order = [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]
        self.difficulty = self.get_ku_difficulty(len(self.graph.nodes), self.topo_order)

        self.students = self.generate_students(student_num)

        self._target = None
        self._legal_candidates = None

        random.seed(None)

    @property
    def mastery(self):
        return [irt(_state, self.difficulty[idx]) for idx, _state in enumerate(self._state)]

    def dump_id2idx(self, filename):
        with open(filename, "w") as wf:
            for node in self.graph.nodes:
                print("%s,%s" % (node, node), file=wf)

    def dump_graph_edges(self, filename):
        with open(filename, "w") as wf:
            for edge in self.graph.edges:
                print("%s,%s" % edge, file=wf)

    def generate_students(self, student_num, step=20):
        student_abilities = self.get_student_ability(student_num)
        students = []

        for student_ability in tqdm(student_abilities, "loading data"):
            self._state = student_ability[:]
            exercises_record = []
            cnt = 0
            if random.random() < ORDER_RATIO:
                while cnt < step:
                    cnt += 1
                    if exercises_record and exercises_record[-1][1] == 1 and len(
                            set([e[0] for e in exercises_record[-3:]])) > 1:
                        for _ in range(1):
                            exercise_record, _, _, _ = self.step(exercises_record[-1][0], interactive=True)
                            exercises_record.append(exercise_record)
                        node = exercises_record[-1][0]
                    elif exercises_record and exercises_record[-1][1] == 0 and random.random() < 0.7:
                        node = exercises_record[-1][0]
                    elif random.random() < 0.9:
                        for node in self.topo_order:
                            if self.mastery[node] < 0.6:
                                break
                        else:
                            break
                    else:
                        node = random.randint(0, len(self.topo_order) - 1)
                    exercise_record, _, _, _ = self.step(node, interactive=True)
                    exercises_record.append(exercise_record)
            else:
                while cnt < step:
                    cnt += 1
                    if random.random() < 0.9:
                        for node in self.default_order:
                            if self.mastery[node] < 0.6:
                                break
                        else:
                            break
                    else:
                        node = random.randint(0, len(self.topo_order) - 1)
                    exercise_record, _, _, _ = self.step(node)
                    exercises_record.append(exercise_record)

            students.append([student_ability, exercises_record,
                             set(random.sample(self.graph.nodes, random.randint(3, len(self.graph.nodes))))])
        return students

    def sim_seq(self, step):
        exercises_record = []
        cnt = 0
        if random.random() < ORDER_RATIO:
            while cnt < step:
                cnt += 1
                if exercises_record and exercises_record[-1][1] == 1 and len(
                        set([e[0] for e in exercises_record[-3:]])) > 1:
                    for _ in range(1):
                        exercise_record = self.step(exercises_record[-1][0])
                        exercises_record.append(exercise_record)
                    node = exercises_record[-1][0]
                elif exercises_record and exercises_record[-1][1] == 0 and random.random() < 0.7:
                    node = exercises_record[-1][0]
                elif random.random() < 0.9:
                    for node in self.topo_order:
                        if self.mastery[node] < 0.6:
                            break
                    else:
                        break
                else:
                    node = random.randint(0, len(self.topo_order) - 1)
                exercise_record = self.step(node)
                exercises_record.append(exercise_record)
        else:
            while cnt < step:
                cnt += 1
                if random.random() < 0.9:
                    for node in self.default_order:
                        if self.mastery[node] < 0.6:
                            break
                    else:
                        break
                else:
                    node = random.randint(0, len(self.topo_order) - 1)
                exercise_record = self.step(node)
                exercises_record.append(exercise_record)
        return exercises_record

    def dump_kt(self, student_num, filename, step=50):
        students = self.get_student_ability(student_num)

        with wf_open(filename) as wf:
            for student in tqdm(students, "kss for kt"):
                self._state = student[:]
                exercises_record = []
                cnt = 0
                if random.random() < ORDER_RATIO:
                    while cnt < step:
                        cnt += 1
                        if exercises_record and exercises_record[-1][1] == 1 and len(
                                set([e[0] for e in exercises_record[-3:]])) > 1:
                            for _ in range(1):
                                exercise_record, _, _, _ = self.step(exercises_record[-1][0])
                                exercises_record.append(exercise_record)
                            node = exercises_record[-1][0]
                        elif exercises_record and exercises_record[-1][1] == 0 and random.random() < 0.7:
                            node = exercises_record[-1][0]
                        elif random.random() < 0.9:
                            for node in self.topo_order:
                                if self.mastery[node] < 0.6:
                                    break
                            else:
                                break
                        else:
                            node = random.randint(0, len(self.topo_order) - 1)
                        exercise_record, _, _, _ = self.step(node)
                        exercises_record.append(exercise_record)
                else:
                    while cnt < step:
                        cnt += 1
                        if random.random() < 0.9:
                            for node in self.default_order:
                                if self.mastery[node] < 0.6:
                                    break
                            else:
                                break
                        else:
                            node = random.randint(0, len(self.topo_order) - 1)
                        exercise_record, _, _, _ = self.step(node)
                        exercises_record.append(exercise_record)
                print(json.dumps(exercises_record), file=wf)

    @property
    def student_num(self):
        return len(self.students)

    @staticmethod
    def get_student_ability(student_num):
        return [[random.randint(-3, 0) - (0.1 * i) for i in range(10)] for _ in range(student_num)]

    @staticmethod
    def get_ku_difficulty(ku_num, order):
        _difficulty = sorted([random.randint(0, 5) for _ in range(ku_num)])
        difficulty = [0] * ku_num
        for index, j in enumerate(order):
            difficulty[j] = _difficulty[index]
        return difficulty

    def state_transform(self, exercise, correct=None):
        graph = self.graph
        a = self._state
        ind = exercise

        if self.path:
            if exercise not in influence_control(graph, a, self.path[-1], allow_shortcut=False, target=self._target,
                                                 legal_candidates=self._legal_candidates)[0]:
                return

        if self.path is not None:
            assert isinstance(exercise, int), exercise
            self.path.append(exercise)

        discount = math.exp(sum([(5 - a[node]) for node in graph.predecessors(ind)] + [0]))
        ratio = 1 / discount
        inc = (5 - a[ind]) * ratio * 0.5

        def _promote(_ind, _inc):
            a[_ind] += _inc
            if a[_ind] > 5:
                a[_ind] = 5
            for node in graph.successors(_ind):
                _promote(node, _inc * 0.5)

        _promote(ind, inc)

    def begin_episode(self):
        while True:
            _idx = random.randint(0, len(self.students) - 1)
            exercises, target = self.get_student(_idx)
            if self.is_valid_sample(target):
                return exercises, target
            else:
                self.remove_invalid_student(_idx)

    def end_episode(self):
        self.path = None
        self._target = None
        self._legal_candidates = None
        self._state = None
        self._initial_state = None

    def get_student(self, idx):
        student = self.students[idx]
        target = student[2]
        self._state = student[0][:]
        self._initial_state = student[0][:]
        self.path = [student[1][-1][0]]
        self._target = set(target)
        self._legal_candidates = set(target)
        return student[1], target
