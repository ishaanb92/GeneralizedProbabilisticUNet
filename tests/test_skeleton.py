# -*- coding: utf-8 -*-

import pytest
from probabilistic_unet.skeleton import fib

__author__ = "kilgore92"
__copyright__ = "kilgore92"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
