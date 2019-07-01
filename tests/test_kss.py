# coding: utf-8
# create by tongshiwei on 2019/6/26

import gym

from EduSim import RandomAgent, Graph


def test_env():
    environment = gym.make("EduSim:KSS-v0")
    environment.render()


def test_interaction():
    step_num = 20
    episode_num = 40
    scores = []

    environment = gym.make("EduSim:KSS-v0")
    agent = RandomAgent(Graph("KSS"))
    for i in range(episode_num):
        # episode
        exercises_record, target = environment.begin_episode()
        initial_score = environment.test_score(target)

        agent.begin_episode(exercises_record, target)

        for j in range(step_num):
            action, q = agent.step()
            assert isinstance(action, int)
            (exercise, correct), _, _, _ = environment.step(exercise=action)

            assert isinstance(exercise, int)
            assert correct in {0, 1}
            agent.state_transform(exercise, correct)

        final_score = environment.test_score(target)
        environment.end_episode()

        scores.append((final_score - initial_score) / len(target))

    print(sum(scores) / episode_num)


def test_eval():
    environment = gym.make("EduSim:KSS-v0")
    agent = RandomAgent(Graph("KSS"))
    print(environment.eval(agent, max_steps=20))
