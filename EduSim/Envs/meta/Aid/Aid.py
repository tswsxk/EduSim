# coding: utf-8
# 2020/4/29 @ tongshiwei


class Aid(object):
    def present(self, learning_item_id=None, *args, **kwargs):
        """present a learning item to Learner"""
        raise NotImplementedError

    def feedback(self, response, *args, **kwargs):
        raise NotImplementedError
