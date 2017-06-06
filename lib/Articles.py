from enum import Enum


class Article(Enum):
    def __str__(self):
        return self.name

    ZERO = '0'
    THE = '1'
    A = '2'
