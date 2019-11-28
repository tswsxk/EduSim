# coding: utf-8
# 2019/11/27 @ tongshiwei

from EduSim.Envs.KSS import KSS
from tqdm import tqdm
import json
from longling import path_append
from EduSim.Envs.base import return_key_env_step, return_key_env_summary_episode


def test_api(tmp_path):
    _env = KSS(learner_num=1, log_rf=path_append(tmp_path, "kss_rf.json", to_str=True))
    _env.remove_invalid_sample(0)
    try:
        with _env.episode():
            pass
    except ValueError:
        pass

    assert True


def test_offline_data(env, tmp_path):
    env.dump_id2idx(path_append(tmp_path, "graph_vertex.json", to_str=True))
    env.dump_graph_edges(path_append(tmp_path, "prerequisite.json", to_str=True))
    exercises_record_for_kt_file = path_append(tmp_path, "kss_kt.json", to_str=True)

    env.dump_kt(10, exercises_record_for_kt_file, 50)

    with open(exercises_record_for_kt_file) as f:
        for line in tqdm(f):
            data = json.loads(line)
            assert len(data) <= 50
            for d in data:
                assert len(d) == 2

    learner = env.begin_episode()
    exercises_record = learner.exercise_history
    assert len(exercises_record) <= 20
    for exercise_record in exercises_record:
        assert len(exercise_record) == 2

    env.end_episode()


def test_env(env):
    with env.episode():
        for learning_item in env.ks.nodes:
            t = env.step(learning_item)
            for key in return_key_env_step:
                assert key in t
            assert {"exercise", "correct"} == set(env.test(learning_item))
            env.step_reward()
        assert set(env.summary_episode()) == set(return_key_env_summary_episode)


def test_interaction(env, agents):
    env.train(agents["random_agent"], max_steps=20, max_episode_num=5)
    env.eval(agents["random_agent"], max_steps=20, max_episode_num=5)
    assert True
