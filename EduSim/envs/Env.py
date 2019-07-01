# coding: utf-8
# create by tongshiwei on 2019/6/25

import json
import logging
import random

import numpy as np
import gym
import networkx as nx
from longling import clock
from longling import flush_print
from longling.lib.candylib import as_list


def get_reward(dataset=None, agent_kind=None):
    return GreedyExpReward()


class Reward(object):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        def global_reward_decay(_global_reward):
            _global_reward = 0
            return _global_reward

        global_reward = delta
        normalize_factor = delta_base

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values

    @staticmethod
    def mature_reward(reward_values):
        reward_values = np.array(reward_values)
        eps = np.finfo(reward_values.dtype).eps.item()
        reward_values = (reward_values - reward_values.mean()) / (reward_values.std() + eps)
        return reward_values


class GreedyExpReward(Reward):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        def global_reward_decay(_global_reward):
            _global_reward *= 0.99
            return _global_reward

        global_reward = delta
        normalize_factor = full_score

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values


class Env(gym.Env):
    """
    Different from traditional game simulation that a certain action can end the game with explicit ending condition,
    learning can last for a long time.

    So, in addition to the basic functions of gym.Env, we add the following methods:

    begin_episode
    end_episode

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        self.reward = None
        self._state = None
        self._initial_state = None
        self.score_for_test = None
        self.logger = logging

    def render(self, mode='human'):
        print(self._initial_state)
        print(self._state)

    def reset(self):
        self.end_episode()
        self.begin_episode()

    def step(self, exercise, **kwargs):
        return self(exercise)[0], None, False, None

    @property
    def mastery(self):
        raise NotImplementedError

    @property
    def state(self):
        return self.mastery

    def begin_episode(self, *args, **kwargs):
        raise NotImplementedError

    def end_episode(self, *args, **kwargs):
        raise NotImplementedError

    def is_valid_sample(self, target):
        if self.test_score(target) == len(target):
            return False
        return True

    def remove_invalid_student(self, idx):
        raise NotImplementedError

    @property
    def student_num(self):
        raise NotImplementedError

    def get_student(self, idx):
        raise NotImplementedError

    def __call__(self, exercises):
        assert self._state
        exercises = as_list(exercises)

        exercises_record = []
        for exercise in exercises:
            assert isinstance(exercise, int)
            correct = self.correct(exercise)
            exercises_record.append((exercise, correct))
            self.state_transform(exercise, correct)
        return exercises_record

    def state_transform(self, exercise, correct=None):
        raise NotImplementedError

    def correct(self, exercise):
        return 1 if random.random() <= self.mastery[exercise] else 0

    def test_correct(self, exercise, mastery):
        return 1 if 0.5 <= mastery[exercise] else 0

    def test(self, exercises, score_type=False, mastery=None):
        mastery = mastery if mastery is not None else self.mastery
        if score_type:
            return [(exercise, mastery[exercise]) for exercise in exercises]
        return [(exercise, self.test_correct(exercise, mastery)) for exercise in exercises]

    def test_score(self, exercises, score_type=None, mastery=None):
        score_type = self.score_for_test if score_type is None else score_type
        return sum([s for _, s in self.test(exercises, score_type, mastery)])

    def eval(self, agent, max_steps, epsilon=None, alternative_mode=True, log_f=None, max_num=None,
             delay_estimation=False, enable_agent_learning=False,
             **kwargs):
        """

        Parameters
        ----------
        agent: Agent
        max_steps: int
        epsilon: None or float, between[0, 1]
        alternative_mode:
        log_f:
        max_num: int or None
        delay_estimation: bool
        enable_agent_learning: bool

        Returns
        -------

        """

        graph = agent.graph

        alpha = 0
        beta = 0
        gamma = 0
        theta = 0

        episode = 0
        _epoch_reward = 0
        _epoch_q_value = 0

        timer = clock.Clock()
        cost_time = 0
        invalid_count = 0

        max_num = self.student_num if max_num is None else max_num

        while True:
            if episode >= self.student_num:
                break

            if max_num is not None and episode >= max_num:
                break
            flush_print(
                "evaluating agent %s | %s, invalid-%s" % (
                    episode + 1, min([max_num, self.student_num]), invalid_count))

            exercises_record, target = self.get_student(episode)

            if not self.is_valid_sample(target):
                self.remove_invalid_student(episode)
                invalid_count += 1
                self.logger.debug("invalid sample: (%s, %s), deleted" % (exercises_record, target))
                continue

            episode += 1

            _episode_exploitation_q_value = 0
            _episode_exploitation_steps = 0

            agent.begin_episode(exercises_record, target, max_steps=max_steps)
            initial_score = self.test_score(target)

            timer.start()
            terminal_tag = False
            if delay_estimation is False:
                for _ in range(max_steps):
                    action, q = agent.step(epsilon=epsilon)

                    if action is None:
                        break

                    if q is not None:
                        _episode_exploitation_q_value += q
                        _episode_exploitation_steps += 1

                    if agent.is_terminal_action(action):
                        terminal_tag = True
                        break

                    if alternative_mode is True:
                        (_, correct), _, _, _ = self.step(exercise=action)
                    else:
                        correct = None

                    agent.state_transform(action, correct, add_to_path=True, reward=None)
                    self.state_transform(action)
                rec_path = agent.path

            else:
                for _ in range(max_steps):
                    agent.step()
                rec_path = agent.end_episode()
                for node in rec_path:
                    self.state_transform(node)
                if len(rec_path) < max_steps:
                    terminal_tag = True

            # 计时结束
            _cost_time = (timer.end() / len(rec_path)) if len(rec_path) > 0 else 0
            cost_time += _cost_time

            # 评价
            full_score = len(target)
            final_score = self.test_score(target)
            reward_values = self.reward(
                initial_score=initial_score,
                final_score=final_score,
                full_score=full_score,
                path=rec_path,
                terminal_tag=terminal_tag,
            )
            if enable_agent_learning is True:
                agent.end_episode(reward_values=reward_values)

            _episode_reward = sum(reward_values)

            # 计算指标
            delta = final_score - initial_score
            delta_base = full_score - initial_score

            _alpha = delta / len(target)
            _beta = _alpha / len(rec_path) if len(rec_path) > 0 else 0
            _gamma = delta / delta_base
            _theta = _gamma / len(rec_path) if len(rec_path) > 0 else 0

            alpha += _alpha
            beta += _beta
            gamma += _gamma
            theta += _theta

            if _episode_exploitation_steps != 0:
                _ave_episode_exploitation_q_value = _episode_exploitation_q_value / _episode_exploitation_steps
                _epoch_q_value += _ave_episode_exploitation_q_value
            else:
                _ave_episode_exploitation_q_value = float('nan')
                _epoch_q_value += 0

            _epoch_reward += _episode_reward

            if log_f is not None:
                _exercises_record = [(graph.idx2id(e[0]), e[1]) for e in exercises_record]
                _target = [graph.idx2id(t) for t in target]
                _rec_path = [graph.idx2id(p) for p in rec_path]
                print(json.dumps({
                    "evaluation": {"alpha": _alpha, "beta": _beta, "gamma": _gamma, "theta": _theta},
                    "target": _target,
                    "target_raw": list(target),
                    "exercises_record": _exercises_record,
                    "exercises_record_raw": exercises_record,
                    "rec_path": _rec_path,
                    "rec_path_raw": rec_path,
                    "episode_ave_q_value": _ave_episode_exploitation_q_value,
                    "episode_reward": _episode_reward,
                    "time per step": _cost_time,
                }), file=log_f)

        valid_count = episode

        self.logger.debug("valid_count: %s, invalid_count: %s" % (valid_count, invalid_count))

        print("")
        return {
            "valid_count": valid_count,
            "alpha": alpha / valid_count,
            "beta": beta / valid_count,
            "gamma": gamma / valid_count,
            "theta": theta / valid_count,
            "ave_q_value": _epoch_q_value / valid_count,
            "ave_reward": _epoch_reward / valid_count,
            "time": cost_time / valid_count
        }


def bfs(graph, mastery, pnode, hop, candidates, soft_candidates, visit_nodes=None, visit_threshold=1,
        allow_shortcut=True):
    """

    Parameters
    ----------
    graph: nx.Digraph
    mastery
    pnode
    hop
    candidates: set()
    soft_candidates: set()
    visit_nodes
    visit_threshold

    Returns
    -------

    """
    assert hop >= 0
    if visit_nodes and visit_nodes.get(pnode, 0) >= visit_threshold:
        return

    if allow_shortcut is False or mastery[pnode] < 0.5:
        candidates.add(pnode)
    else:
        soft_candidates.add(pnode)

    if hop == 0:
        return

    # 向前搜索
    for node in list(graph.predecessors(pnode)):
        if allow_shortcut is False or mastery[node] < 0.5:
            bfs(
                graph=graph,
                mastery=mastery,
                pnode=node,
                hop=hop - 1,
                candidates=candidates,
                soft_candidates=soft_candidates,
                visit_nodes=visit_nodes,
                visit_threshold=visit_threshold,
                allow_shortcut=allow_shortcut,
            )

    # 向后搜索
    for node in list(graph.successors(pnode)):
        if visit_nodes and visit_nodes.get(node, 0) >= visit_threshold:
            continue
        if allow_shortcut is False or mastery[node] < 0.5:
            candidates.add(node)
        else:
            soft_candidates.add(node)


def influence_control(graph, mastery, pnode, visit_nodes=None, visit_threshold=1, allow_shortcut=True, no_pre=None,
                      connected_graph=None, target=None, legal_candidates=None, path_table=None):
    """

    Parameters
    ----------
    graph: nx.Digraph
    mastery: list(float)
    pnode: None or int
    visit_nodes: None or dict
    visit_threshold: int
    allow_shortcut: bool
    no_pre: set
    connected_graph: dict
    target: set or list
    legal_candidates: set or None
    path_table: dict or None

    Returns
    -------

    """
    assert pnode is None or isinstance(pnode, int), pnode

    if mastery is None:
        allow_shortcut = False

    # select candidates
    candidates = []
    soft_candidates = []

    if allow_shortcut is True:
        # 允许通过捷径绕过已掌握的点

        # 在已有前驱节点前提下，如果当前节点已经掌握，那么开始学习它的后继未掌握节点
        if pnode is not None and mastery[pnode] >= 0.5:
            for candidate in list(graph.successors(pnode)):
                if visit_nodes and visit_nodes.get(candidate, 0) >= visit_threshold:
                    continue
                if mastery[candidate] < 0.5:
                    candidates.append(candidate)
                else:
                    soft_candidates.append(candidate)
            if candidates:
                return candidates, soft_candidates

        # 否则(即当前节点未掌握), 选取其2跳前驱点及所有前驱点的后继点（未掌握的）作为候选集
        elif pnode is not None:
            _candidates = set()
            _soft_candidates = set()
            for node in list(graph.predecessors(pnode)):
                bfs(graph, mastery, node, 2, _candidates, _soft_candidates, visit_nodes, visit_threshold,
                    allow_shortcut)
            return list(_candidates) + [pnode], list(_soft_candidates)

        # 如果前两种方法都没有选取到候选集，那么进行重新选取
        for node in graph.nodes:
            if visit_nodes and visit_nodes.get(node, 0) >= visit_threshold:
                # 当前结点频繁访问
                continue

            if mastery[node] >= 0.5:
                # 当前结点已掌握，跳过
                soft_candidates.append(node)
                continue

            # 当前结点未掌握，且其前置点都掌握了的情况下，加入候选集
            pre_nodes = list(graph.predecessors(node))
            for n in pre_nodes:
                pre_mastery = mastery[n]
                if pre_mastery < 0.5:
                    soft_candidates.append(node)
                    break
            else:
                candidates.append(node)
    else:
        # allow_shortcut is False
        # 不允许通过捷径绕过已掌握的点
        candidates = set()
        soft_candidates = set()
        if pnode is not None:
            # 加入所有后继点
            candidates = set(list(graph.successors(pnode)))

            if not graph.predecessors(pnode) or not graph.successors(pnode):
                # 没有前驱点 或 没有后继点
                candidates = set(no_pre)

            # 选取其2跳前驱点及所有前驱点的后继点
            for node in list(graph.predecessors(pnode)):
                bfs(graph, mastery, node, 1, candidates, soft_candidates, visit_nodes, visit_threshold, allow_shortcut)

            # 避免死循环
            if candidates:
                candidates.add(pnode)

            # 频繁访问节点过滤
            if visit_nodes:
                candidates -= set([node for node, count in visit_nodes.items() if count >= visit_threshold])

            candidates = list(candidates)

    if not candidates:
        # 规则没有选取到合适候选集
        candidates = list(graph.nodes)
        soft_candidates = list()

    if connected_graph is not None and pnode is not None:
        # 保证候选集和pnode在同一个连通子图内
        candidates = list(set(candidates) & connected_graph[pnode])

    if target is not None and legal_candidates is not None:
        assert target
        # 保证节点可达目标点
        _candidates = set(candidates) - legal_candidates
        for candidate in _candidates:
            if candidate in legal_candidates:
                continue
            for t in target:
                if path_table is not None:
                    if t in path_table[candidate]:
                        legal_tag = True
                    else:
                        legal_tag = False
                else:
                    legal_tag = nx.has_path(graph, candidate, t)
                if legal_tag is True:
                    legal_candidates.add(candidate)
                    break
        candidates = set(candidates) & legal_candidates
        if not candidates:
            candidates = target

    return list(candidates), list(soft_candidates)
