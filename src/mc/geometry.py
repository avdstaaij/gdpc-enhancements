"""
Wrappers around gdpc v5.0 that work with vectors
"""

from typing import Optional, Union, List
from dataclasses import dataclass
from glm import ivec3, bvec2
from gdpc import geometry

from .vectorUtil import FORWARD, LEFT, Rect, Area, addY, areaBetween, Transform
from .block import Block
from .interface import Interface


# An explanation of why this class is here:
#
# We wrap gdpc.interface with our own Interface class. We have several reasons for doing so, most
# importantly to work with vectors and transforms. However, we also want to use some of
# gdpc.geometry's functions to implement the vector-based geometry functions in this file, rather
# than re-inventing the wheel. The functions from gdpc.geometry use gdpc.interface directly, so they
# do not go through our wrapping Interface class.
#
# Initially, this was not a problem: we would just apply the transform of the passed Interface
# manually and then call the functions from gdpc.geometry using the pre-transformed coordinates,
# pre-stringified block and the Interface's underlying gdpc.interface.
#
# However, to implement some advanced features, we need to have a single point - under our control -
# where all block placements pass through. That is, we want all block placements to go through
# Interface. To achieve this without having to re-implement the functions from gdpc.geometry that we
# use, we use this class. What this class does, is act as if it is a gdpc.interface class for all
# purposes needed by the gdpc.geometry functions (duck typing), while it actually redirects all
# placeBlock calls to Interface.
#
# For this to work, we rely on internal implementation details of gdpc. Therefore, it is essential
# that a specific version of gdpc is used (the one in requirements.txt).
#
# For performance reasons, we still manually apply the transform of the passed Interface and
# manually stringify blocks in the wrapping geometry functions. This way, we only have to do so for
# the corner blocks of the geometrical areas, rather than all of them.
@dataclass
class __HackedGDPCInterface():
    interface: Interface

    def placeBlock(self, x, y, z, block, replace=None, doBlockUpdates=-1, customFlags=-1):
        self.interface.placeStringGlobal(ivec3(x, y, z), block[0], block[1], block[2], block[3], replace)
        return "0" # Indicates "no error" to gdpc.

    def setBuffering(self, value, notify):
        pass

    def isBuffering(self): # pylint: disable=no-self-use
        return True # No need to speak the truth here.


def placeLine(itf: Interface, first: ivec3, last: ivec3, block: Block, rotation: int = 0, replace: Optional[Union[str, List[str]]] = None):
    globalFirst = itf.transform * first
    globalLast  = itf.transform * last
    globalRotation = (rotation + itf.transform.rotation) % 4 # We cannot interpolate rotations, so we take only one
    globalScale    = itf.transform.scale # We could take a local scale parameter, but would we really ever use it?
    geometry.placeLine(
        globalFirst.x, globalFirst.y, globalFirst.z,
        globalLast .x, globalLast .y, globalLast .z,
        (globalScale, block.name, block.blockStateString(globalRotation, bvec2(globalScale.x < 0, globalScale.z < 0)), block.nbt),
        replace,
        interface = __HackedGDPCInterface(itf)
    )


def placeCuboid(itf: Interface, first: ivec3, last: ivec3, block: Block, rotation: int = 0, replace: Optional[Union[str, List[str]]] = None, hollow: bool = False):
    globalFirst = itf.transform * first
    globalLast  = itf.transform * last
    globalRotation = (rotation + itf.transform.rotation) % 4 # We cannot interpolate rotations, so we take only one
    globalScale    = itf.transform.scale # We could take a local scale parameter, but would we really ever use it?
    geometry.placeCuboid(
        globalFirst.x, globalFirst.y, globalFirst.z,
        globalLast .x, globalLast .y, globalLast .z,
        (globalScale, block.name, block.blockStateString(globalRotation, bvec2(globalScale.x < 0, globalScale.z < 0)), block.nbt),
        replace, hollow,
        interface = __HackedGDPCInterface(itf)
    )


def placeArea(itf: Interface, area: Area, block: Block, rotation: int = 0, replace: Optional[Union[str, List[str]]] = None, hollow: bool = False):
    if (area.size.x == 0 or area.size.y == 0 or area.size.z == 0): return
    placeCuboid(itf, area.begin, area.end - 1, block, rotation, replace, hollow)


def placeCornerPillars(itf, area, block):
    pillar = Area(area.begin, ivec3(1,area.size.y,1))
    placeArea(itf, pillar, block)
    pillar.begin += LEFT * (area.size.x-1)
    placeArea(itf, pillar, block)
    pillar.begin += FORWARD * (area.size.z-1)
    placeArea(itf, pillar, block)
    pillar.begin -= LEFT * (area.size.x-1)
    placeArea(itf, pillar, block)


def placeRect(itf: Interface, rect: Rect, y: int, block: Block, rotation: int = 0, replace: Optional[Union[str, List[str]]] = None, hollow: bool = False):
    if (rect.size.x == 0 or rect.size.y == 0): return
    placeArea(itf, rect.toArea(y, 1), block, rotation, replace, hollow)


# TODO: improve?
def placeList(itf: Interface, block_list: List[ivec3], block: Block, rotation: int = 0, replace: Optional[Union[str, List[str]]] = None):
    for block_pos in block_list:
        itf.placeBlock(Transform(block_pos, rotation), block, replace)


def placeRectOutline(itf: Interface, rect: Rect, y: int, block: Block, rotation: int = 0, replace: Optional[Union[str, List[str]]] = None):
    transform = Transform(addY(rect.offset, y))
    itf.transform.push(transform)
    placeLine(itf, first=ivec3(0, 0, 0), last=ivec3(rect.size.x - 1, 0, 0),
              block=block, rotation=rotation, replace=replace)
    placeLine(itf, first=ivec3(rect.size.x - 1, 0, 0), last=ivec3(rect.size.x - 1, 0, rect.size.y - 1),
              block=block, rotation=rotation, replace=replace)
    placeLine(itf, first=ivec3(rect.size.x - 1, 0, rect.size.y - 1), last=ivec3(0, 0, rect.size.y - 1),
              block=block, rotation=rotation, replace=replace)
    placeLine(itf, first=ivec3(0, 0, rect.size.y - 1), last=ivec3(0, 0, 0),
              block=block, rotation=rotation, replace=replace)
    itf.transform.pop(transform)


def placeCheckeredCuboid(itf: Interface, first: ivec3, last: ivec3, block1: Block, block2: Block = Block("minecraft:air"), rotation: int = 0, replace: Optional[Union[str, List[str]]] = None):
    placeCheckeredArea(itf, areaBetween(first, last), block1, block2, rotation, replace)


def placeCheckeredArea(itf: Interface, area: Area, block1: Block, block2: Block = Block("minecraft:air"), rotation: int = 0, replace: Optional[Union[str, List[str]]] = None):
    # TODO: simplify this loop
    i = 0
    for x in range(area.begin.x, area.end.x):
        i = (i + 1) % 2
        j = i
        for y in range(area.begin.y, area.end.y):
            j = (j + 1) % 2
            k = j
            for z in range(area.begin.z, area.end.z):
                k = (k + 1) % 2
                itf.placeBlock(Transform(ivec3(x,y,z), rotation), block1 if k == 0 else block2, replace)


def placeZStripedArea(itf: Interface, area: Area, block1: Block, block2: Block = Block("minecraft:air"), rotation: int = 0, replace: Optional[Union[str, List[str]]] = None):
    i = 0
    for x in range(area.begin.x, area.end.x):
        i = (i + 1) % 2
        j = i
        for y in range(area.begin.y, area.end.y):
            # j = (j + 1) % 2
            k = j
            for z in range(area.begin.z, area.end.z):
                # k = (k + 1) % 2
                itf.placeBlock(Transform(ivec3(x,y,z), rotation), block1 if k == 0 else block2, replace)


def placeXStripedArea(itf: Interface, area: Area, block1: Block, block2: Block = Block("minecraft:air"), rotation: int = 0, replace: Optional[Union[str, List[str]]] = None):
    i = 0
    for x in range(area.begin.x, area.end.x):
        # i = (i + 1) % 2
        j = i
        for y in range(area.begin.y, area.end.y):
            # j = (j + 1) % 2
            k = j
            for z in range(area.begin.z, area.end.z):
                k = (k + 1) % 2
                itf.placeBlock(Transform(ivec3(x,y,z), rotation), block1 if k == 0 else block2, replace)
