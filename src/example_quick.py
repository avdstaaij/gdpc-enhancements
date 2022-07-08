#!/usr/bin/env python3

""" The quick example from the README.

    See example.py for a fuller example.
"""

__author__    = "Arthur van der Staaij"
__copyright__ = "Copyright 2022, Arthur van der Staaij"
__licence__   = "MIT"


from glm import ivec3

from gdpce.vector_util import Box
from gdpce.transform import Transform
from gdpce.block import Block
from gdpce.interface import Interface, getBuildArea
from gdpce.geometry import placeBox

import models


buildArea = getBuildArea()

itf = Interface(buildArea.offset)

# Place a block
itf.placeBlock(ivec3(1,0,1), Block("grass_block"))

# Build a cube
placeBox(itf, Box(ivec3(4,0,0), ivec3(3,3,3)), Block("stone"))

# Build an oriented building in transformed local coordinates
transform = Transform(translation=ivec3(10,0,0), rotation=1, scale=ivec3(1,2,1))
with itf.pushTransform(transform):
    placeBox(itf, Box(size=ivec3(3,1,3)), Block("oak_log", axis="x"))

# Place a block with NBT data
nbt = 'Items: [{Slot: 13, id: "apple", Count: 1}]'
itf.placeBlock(ivec3(13,0,1), Block("chest", facing="south", nbt=nbt))

# Build a saved model, with any position and orientation, and with substitutions
models.testShape.build(
    itf,
    Transform(ivec3(16,0,0), rotation=3, scale=ivec3(1,1,-1)),
    substitutions={
        "minecraft:oak_planks": ["acacia_planks", "dark_oak_planks"]
    }
)
