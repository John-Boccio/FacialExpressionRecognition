"""
Author(s):
    John Boccio
Last revision:
    10/18/2019
Description:
    Standard facial expression identifier to be used throughout the project so we can stay consistent between different
    databases.
"""
from enum import Enum


class Expression(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    SAD = 4
    SURPRISE = 5
    NEUTRAL = 6
