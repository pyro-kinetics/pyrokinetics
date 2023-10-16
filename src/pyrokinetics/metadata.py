import uuid
from datetime import datetime
from typing import Dict

try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version("pyrokinetics")
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except ImportError:
        __version__ = "0.0.1"


# Define UUID and session start as module-level variables.
# Determined at the first import, and should be fixed during each session.

__session_uuid = uuid.uuid4()
__session_start = datetime.now()


def metadata(title: str, obj: str, **kwargs: str) -> Dict[str, str]:
    """
    Return a dict of metadata which can be used to uniquely identify Pyrokinetics
    objects. Should be written to Pyrokinetics output files to establish their
    provenance.

    Parameters
    ----------
    title: str
        A title to be assigned to the object. Recommended to use the object name if
        no other obvious title can be determined.
    obj: str
        The type of Pyrokinetics object, expressed as a string. It is recommended to
        use ``self.__class__.__name__`` in external classes.
    **kwargs: str
        Extra keywords to add to the returned dict.
    """
    return {
        "title": str(title),
        "software_name": "Pyrokinetics",
        "software_version": __version__,
        "object_type": str(obj),
        "object_uuid": str(uuid.uuid4()),
        "object_created": str(datetime.now()),
        "session_uuid": str(__session_uuid),
        "session_started": str(__session_start),
        **kwargs,
    }
