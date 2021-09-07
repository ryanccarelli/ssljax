import pytest

def test_ssltrainer():
    task = Task(cputestconfig)
    trainer = task.trainer
    trainer.train()
