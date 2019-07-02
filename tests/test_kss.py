# coding: utf-8
# create by tongshiwei on 2019/6/26

import json

import gym
from tqdm import tqdm

from EduSim import RandomAgent, Graph


def test_env(tmp_path):
    environment = gym.make("EduSim:KSS-v0")
    environment.render()

    exercises_record_for_kt_file = str(tmp_path / "kss-kt.json")
    environment.dump_kt(4000, exercises_record_for_kt_file, 50)

    with open(exercises_record_for_kt_file) as f:
        for line in tqdm(f):
            data = json.loads(line)
            assert len(data) <= 50
            for d in data:
                assert len(d) == 2

    students = environment.generate_students(4000, 20)
    for exercises_record in [student[1] for student in students]:
        assert len(exercises_record) <= 20
        for exercise_record in exercises_record:
            assert len(exercise_record) == 2


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
    assert True
