# coding: utf-8
# 2019/11/27 @ tongshiwei

def test_q_bank(q_bank):
    assert isinstance(q_bank.get_difficulty(3), (int, float))
