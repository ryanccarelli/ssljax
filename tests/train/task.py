import parameterized
import pytest
from ssljax.train.task import Task


class TaskTest:
    def setUp(self, cputestconfig):
        super().setUp()
        # create a cputestconfig
        self.cfg = cputestconfig

    def test_init(self):
        # declare the most minimal task
        task = Task(self.cfg)
