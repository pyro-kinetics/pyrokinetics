from pyrokinetics.kinetics import KineticsReader, KineticsReaderFactory
from inspect import isclass
from os import remove
import pytest


class MyReader(KineticsReader):
    """Defines a minimal concrete class for KineticsReader"""

    def read(self, filename):
        f = open(filename, "r")
        result = f.read()
        f.close()
        return result


class TestKineticsReaderFactory:
    @pytest.fixture
    def example_input_file(self):
        f = open("example_input_file.txt", "w")
        f.write("hello world")
        yield
        f.close()
        remove("example_input_file.txt")

    @pytest.fixture
    def kinetics_reader_factory(self, example_input_file):
        factory = KineticsReaderFactory()
        factory["MyReader"] = MyReader
        return factory

    def test_registering(self, kinetics_reader_factory):
        """Test that a KineticsReader has been successfully registered, and that they
        can be accessed like the keys of a dict.
        """
        assert "MyReader" in kinetics_reader_factory
        assert "MyReader" in kinetics_reader_factory.keys()

    @pytest.mark.parametrize(
        "key,value",
        [
            ("MyReader", MyReader()),  # test with instance rather than class
            ("MyReader", str),  # test with non-KineticsReader
            ("MyReader", 17),  # test with unrelated object
        ],
    )
    def test_registering_bad_inputs(self, kinetics_reader_factory, key, value):
        """Test that KineticsReaderFactory rejects bad inputs"""
        with pytest.raises(TypeError) as excinfo:
            kinetics_reader_factory[key] = value
        if isclass(value):
            assert "subclass KineticsReader" in str(excinfo.value)
        else:
            assert "Only classes" in str(excinfo.value)

    def test_creating_reader(self, kinetics_reader_factory):
        reader = kinetics_reader_factory["MyReader"]
        assert isinstance(reader, KineticsReader)
        assert isinstance(reader, MyReader)

    def test_bad_key(self, kinetics_reader_factory):
        with pytest.raises(KeyError) as excinfo:
            reader = kinetics_reader_factory["OtherReader"]
        assert "OtherReader" in str(excinfo.value)

    def test_infer_type(self, kinetics_reader_factory):
        reader = kinetics_reader_factory["example_input_file.txt"]
        assert isinstance(reader, MyReader)

    def test_pop(self, kinetics_reader_factory):
        # register a duplicate to make sure we don't break everything...
        kinetics_reader_factory["MyOtherReader"] = MyReader
        assert len(kinetics_reader_factory) == 2
        kinetics_reader_factory.pop("MyOtherReader")
        assert "MyOtherReader" not in kinetics_reader_factory
        assert len(kinetics_reader_factory) == 1
