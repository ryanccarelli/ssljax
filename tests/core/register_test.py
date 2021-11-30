import pytest
from ssljax.core import register, get_from_register

class TestRegister:
    def test_add_get_from_register(self):
        class DummyClass:
            pass

        @register(DummyClass, "dummyinstance")
        class dummyinstance(DummyClass):
            pass

        @register(DummyClass, "dummyfunction")
        def dummyfunction():
            return True

        assert issubclass(get_from_register(DummyClass, "dummyinstance"), DummyClass)
        assert(get_from_register(DummyClass, "dummyfunction")())
