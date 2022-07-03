from dataclasses import dataclass
from typing import Union, Optional, List, Dict
from copy import copy
from glm import ivec3

from util.util import filledList
from .util import to_namespaced_id
from .vectorUtil import Transform, Area
from .interface import Interface
from .block import Block


@dataclass
class Model:
    def __init__(self, size: ivec3, blocks: Optional[List[List[List[Optional[Block]]]]] = None):
        self.size = size
        if blocks is not None:
            self.blocks = blocks
        else:
            self.blocks = filledList(size.x, filledList(size.y, filledList(size.z, None)))

    size:   ivec3
    blocks: List[List[List[Optional[Block]]]]

    def block(self, vec: ivec3):
        return self.blocks[vec.x][vec.y][vec.z]

    def build(self, itf: Interface, t = Transform(), substitutions: Optional[Dict[str, str]] = None, replace: Optional[Union[str, List[str]]] = None):
        if substitutions is None: substitutions = {}

        @dataclass
        class LateBlockInfo:
            block:    Block
            position: ivec3

        late_blocks: List[LateBlockInfo] = []

        itf.transform.push(t)

        for vec in Area(size=self.size).inner:
            block = self.blocks[vec.x][vec.y][vec.z]
            if block is not None:
                blockToPlace = copy(block)
                blockToPlace.name = substitutions.get(block.name, block.name)
                if blockToPlace.needs_late_placement:
                    late_blocks.append(LateBlockInfo(blockToPlace, vec))
                else:
                    itf.placeBlock(vec, blockToPlace, replace)

        # Place the late blocks, thrice.
        # Yes, placing them three time is really necessary. Wall-type blocks require it.
        for late_block_info in late_blocks:
            itf.placeBlock(
                late_block_info.position,
                late_block_info.block,
                replace
            )
        for late_block_info in late_blocks[::-1]:
            itf.placeBlock(
                late_block_info.position,
                late_block_info.block,
                (None if replace is None else to_namespaced_id(late_block_info.block.name))
            )
        for late_block_info in late_blocks:
            itf.placeBlock(
                late_block_info.position,
                late_block_info.block,
                (None if replace is None else to_namespaced_id(late_block_info.block.name))
            )

        itf.transform.pop(t)
