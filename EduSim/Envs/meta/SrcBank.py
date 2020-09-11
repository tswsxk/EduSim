# coding: utf-8
# 2020/4/29 @ tongshiwei


class SrcBank(object):
    def __getitem__(self, item):
        """
        Examples
        --------
        >>> bank = SrcBank()
        >>> bank[123]
        123
        """
        return item


class EBank(SrcBank):
    """Exercise/Question Bank"""
    pass


class MBank(SrcBank):
    """Material Bank"""
    pass
