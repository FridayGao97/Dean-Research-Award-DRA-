from .CAD120.cad120 import CAD120
from .HICO.hico import HICO
# from .VCOCO.vcoco import VCOCO

import utils
from .CAD120.cad120 import metadata as cad_metadata
from .HICO import metadata as hico_metadata
# import VCOCO.metadata as vcoco_metadata

__all__ = ('CAD120', 'HICO', 'utils', 'cad_metadata', 'hico_metadata')
