import uuid
from datetime import datetime
from typing import Dict

try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except ImportError:
        __version__ = "0.0.1"


# Define UUID and session start as module-level variables.
# Determined at the first import, and should be fixed during each session.

__uuid = uuid.uuid4()
__session_start = datetime.now()


def metadata(**kwargs: str) -> Dict[str, str]:
    """
    Return a dict of metadata uniquely describing this Pyrokinetics session. Should be
    written to Pyrokinetics output files to establish their provenance.

    Parameters
    ----------
    **kwargs: str
        Extra keywords to add to the returned dict.
    """
    return {
        "software_name": "Pyrokinetics",
        "software_version": __version__,
        "session_started": str(__session_start),
        "session_uuid": str(__uuid),
        "date_created": str(datetime.now()),
        **kwargs,
    }
