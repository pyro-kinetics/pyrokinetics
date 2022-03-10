# Import GKCode classes
from .GKCode import GKCode, gk_codes
from .GKCodeGS2 import GKCodeGS2
from .GKCodeGENE import GKCodeGENE
from .GKCodeCGYRO import GKCodeCGYRO

# Register each reader type with factory
gk_codes["GS2"] = GKCodeGS2
gk_codes["GENE"] = GKCodeGENE
gk_codes["CGYRO"] = GKCodeCGYRO

from .GKOutput import GKOutput

# Import refactored versions
from .GKInputReader import GKInputReader, gk_input_readers
from .GKInputWriter import GKInputWriter, gk_input_writers
from .GKOutputReader import GKOutputReader, gk_output_readers
from .GKInputReaderGS2 import GKInputReaderGS2
from .GKInputWriterGS2 import GKInputWriterGS2
from .GKOutputReaderGS2 import GKOutputReaderGS2

gk_input_readers["GS2"] = GKInputReaderGS2
gk_input_writers["GS2"] = GKInputWriterGS2
gk_output_readers["GS2"] = GKOutputReaderGS2


__all__ = ["GKCode", "GKOutput", "GKInputReader", "GKInputWriter", "GKOutputReader"]
