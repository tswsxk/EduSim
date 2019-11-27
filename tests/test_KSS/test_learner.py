# coding: utf-8
# 2019/11/27 @ tongshiwei


def test_learner(learner):
    initial_ability = learner.test(0)
    assert isinstance(initial_ability, (int, float))
    assert initial_ability == learner.test(0)
    for i in learner.target:
        learner.learn(i)
    assert initial_ability != learner.test(0)
