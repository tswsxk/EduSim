# coding: utf-8
# 2019/11/25 @ tongshiwei

import pytest
from .conftest import IrtLearner


@pytest.fixture(params=[None, "random"])
def test_irt_learner(stu_params):
    """
    To test:
    * state changes after learning
    * state does not change after testing
    """
    concept_num = 10
    student = IrtLearner(concept_num, stu_params.params)
    init_state = student.state_snapshot()
    student.learn(3)
    state1 = student.state_snapshot()
    assert init_state[3] != state1[3]
    student.test(3)
    state2 = student.state_snapshot()
    assert state1[3] == state2[3]
