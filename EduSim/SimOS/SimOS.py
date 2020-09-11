# coding: utf-8
# 2020/4/30 @ tongshiwei

import random
import logging

from EduSim.Envs.meta.Env import Env
from .config import as_level
from longling.ML.toolkit.monitor import ConsoleProgressMonitor, EMAValue


class MetaAgent(object):
    def begin_episode(self, *args, **kwargs):
        raise NotImplementedError

    def end_episode(self, observation, reward, done, info):
        raise NotImplementedError

    def observe(self, observation, reward, done, info):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def tune(self, *args, **kwargs):
        raise NotImplementedError

    def n_step(self, max_steps: int):
        return []


class RandomAgent(MetaAgent):
    def __init__(self, material_nums):
        self._n = material_nums

    def begin_episode(self, *args, **kwargs):
        pass

    def end_episode(self, observation, reward, done, info):
        pass

    def observe(self, observation, reward, done, info):
        pass

    def step(self):
        return random.randint(0, self._n - 1)

    def n_step(self, max_steps: int):
        return [random.randint(0, self._n - 1) for _ in range(max_steps)]

    def tune(self, *args, **kwargs):
        pass


def train_eval(agent: MetaAgent, env: Env, max_steps: int = None, max_episode_num: int = None, n_step=False,
               train=False,
               logger=logging, level="episode", episode_callback=None, summary_callback=None):
    episode = 0

    level = as_level(level)

    rewards = []
    infos = []
    values = {"Episode": EMAValue(["Reward"])}

    if level >= as_level("summary"):
        monitor = ConsoleProgressMonitor(
            indexes={"Episode": ["Reward"]},
            values=values,
            total=max_episode_num,
            player_type="episode"
        )
        monitor.player.start()
    else:
        monitor = None

    while True:
        if max_episode_num is not None and episode >= max_episode_num:
            break

        try:
            learner_profile = env.begin_episode()
            agent.begin_episode(learner_profile)
            episode += 1

            if level <= as_level("episode"):
                logger.info("episode [%s]: %s" % (episode, env.render("log")))

        except ValueError:  # pragma: no cover
            break

        # recommend and learn
        if n_step is True:
            assert max_steps is not None
            # generate a learning path
            learning_path = agent.n_step(max_steps)
            env.n_step(learning_path)
        else:
            learning_path = []
            _step = 0

            if max_steps is not None:
                # generate a learning path step by step
                for _ in range(max_steps):
                    try:
                        learning_item = agent.step()
                        learning_path.append(learning_item)
                    except ValueError:  # pragma: no cover
                        break
                    observation, reward, done, info = env.step(learning_item)

                    if level <= as_level("step"):
                        _step += 1
                        logger.debug(
                            "step [%s]: agent -|%s|-> env, env state %s" % (_step, learning_item, env.render("log"))
                        )
                        logger.debug(
                            "step [%s]: observation: %s, reward: %s" % (_step, observation, reward)
                        )

                    agent.observe(observation, reward, done, info)
                    if done:
                        break
            else:
                while True:
                    learning_item = agent.step()
                    observation, reward, done, info = env.step(learning_item)
                    learning_path.append(learning_item)
                    agent.observe(observation, reward, done, info)
                    if done:
                        break

        # test the learner to see the learning effectiveness
        observation, reward, done, info = env.end_episode()
        agent.end_episode(observation, reward, done, info)

        rewards.append(reward)
        infos.append(info)

        if level <= as_level("episode"):
            logger.info("episode [%s] - learning path: %s" % (episode, learning_path))
            logger.info("episode [%s] - total reward: %s" % (episode, reward))
            logger.info("episode [%s]: %s" % (episode, env.render(mode="log")))

        elif monitor is not None:
            values["Episode"]("Reward", reward)
            monitor.player(episode)

        env.reset()

        if episode_callback is not None:
            episode_callback(episode, reward, done, info, logger)

        if train is True:
            agent.tune()

    if summary_callback is not None and level <= as_level("summary"):
        return summary_callback(rewards, infos, logger)

    if monitor is not None:
        monitor.player.end()

    return rewards, infos
