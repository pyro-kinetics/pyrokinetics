from pyrokinetics.readers import Reader, create_reader_factory
from inspect import isclass
from os import remove
import pytest


class MyReader(Reader):
    """Defines a minimal concrete class for Reader"""

    def read(self, filename):
        f = open(filename, "r")
        result = f.read()
        f.close()
        return result


class TestReaderFactory:
    @pytest.fixture
    def example_input_file(self):
        f = open("example_input_file.txt", "w")
        f.write("hello world")
        yield
        f.close()
        remove("example_input_file.txt")

    @pytest.fixture
    def reader_factory(self, example_input_file):
        factory = create_reader_factory()
        factory["MyReader"] = MyReader
        return factory

    def test_autonaming(self, reader_factory):
        """The default name of the reader_factory class should be ReaderFactory"""
        assert reader_factory.__class__.__name__ == "ReaderFactory"

    def test_registering(self, reader_factory):
        """Test that a Reader has been successfully registered, and that they
        can be accessed like the keys of a dict.
        """
        assert "MyReader" in reader_factory
        assert "MyReader" in reader_factory.keys()

    @pytest.mark.parametrize(
        "key,value",
        [
            ("MyReader", MyReader()),  # test with instance rather than class
            ("MyReader", str),  # test with non-Reader
            ("MyReader", 17),  # test with unrelated object
        ],
    )
    def test_registering_bad_inputs(self, reader_factory, key, value):
        """Test that ReaderFactory rejects bad inputs"""
        with pytest.raises(TypeError) as excinfo:
            reader_factory[key] = value
        if isclass(value):
            assert "subclass Reader" in str(excinfo.value)
        else:
            assert "Only classes" in str(excinfo.value)

    def test_creating_reader(self, reader_factory):
        reader = reader_factory["MyReader"]
        assert isinstance(reader, Reader)
        assert isinstance(reader, MyReader)

    def test_bad_key(self, reader_factory):
        with pytest.raises(KeyError) as excinfo:
            reader_factory["OtherReader"]
        assert "OtherReader" in str(excinfo.value)

    def test_infer_type(self, reader_factory):
        reader = reader_factory["example_input_file.txt"]
        assert isinstance(reader, MyReader)

    def test_pop(self, reader_factory):
        # register a duplicate to make sure we don't break everything...
        reader_factory["MyOtherReader"] = MyReader
        assert len(reader_factory) == 2
        reader_factory.pop("MyOtherReader")
        assert "MyOtherReader" not in reader_factory
        assert len(reader_factory) == 1
