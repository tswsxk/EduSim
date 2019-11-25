# coding: utf-8
# 2019/11/25 @ tongshiwei

__all__ = ["SimStu"]


class SimStu(object):
    def learn(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError
