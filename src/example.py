#!/usr/bin/env python3

""" Executable script that build an example structure in the center of the build area.

    The purpose of this file is to demonstrate the usage of the core GDPC v5.0 enhancements provided
    by this repository. It does not demonstrate every enhancement, and it assumes some familiarity
    with the base GDPC framework.

    The following enhancements are demonstrated:
    - Vectors and various vector math utilities.
    - The Transform class and the use of local coordinate systems.
    - The placement of blocks with NBT data.
    - The placement of saved models.

    To install or learn more about the base GDPC framework, visit
    https://github.com/nilsgawlik/gdmc_http_client_python
"""

__author__    = "Arthur van der Staaij"
__copyright__ = "Copyright 2022, Arthur van der Staaij"
__licence__   = "MIT"


import sys

import numpy as np
from glm import ivec2, ivec3, bvec3

from util.util import eprint

from mc.vector_util import addY, vecString, Rect, Box, centeredSubRect, rectSlice
from mc.transform import Transform, rotatedBoxTransform, scaledBoxTransform, flippedBoxTransform
from mc.nbt_util import signNBT
from mc.block import Block
from mc.interface import Interface, getBuildArea, getWorldSlice
from mc.geometry import placeBox, placeRectOutline, placeCheckeredBox

import models


EXAMPLE_STRUCTURE_SIZE = ivec2(29, 14)


# You could take a Transform parameter here and push it on the callee side, or you could require
# that this is done on the caller side. In this example, we choose the latter.
def buildExampleStructure(itf: Interface):
    # Clear the area
    placeBox(itf, Box(size=addY(EXAMPLE_STRUCTURE_SIZE, 10)), Block("air"))

    # Build a checkered floor
    placeCheckeredBox(itf, Box(size=addY(EXAMPLE_STRUCTURE_SIZE, 1)), Block("gray_concrete"), Block("light_gray_concrete"))

    # Place a block!
    itf.placeBlock(ivec3(2,1,2), Block("grass_block"))

    # Build a cube
    placeBox(itf, Box(ivec3(5,1,1), ivec3(3,3,3)), Block("stone"))

    # Build a textured cube
    placeBox(itf, Box(ivec3(9,1,1), ivec3(3,3,3)), Block(["stone", "andesite", "cobblestone"]))

    # Overwrite the textured cube; block palettes can contain empty strings, meaning "no placement"
    placeBox(itf, Box(ivec3(9,1,1), ivec3(3,3,3)), Block(["mossy_cobblestone"] + 5*[""]))

    # Build a cube in local coordinates
    transform = Transform(ivec3(13,1,1))
    itf.transform.push(transform)
    placeBox(itf, Box(size=ivec3(3,3,3)), Block("oak_planks"))
    itf.transform.pop(transform)

    # Avoid the need to call Transform.pop() with Interface.pushTransform()
    with itf.pushTransform(): # Reverts changes to itf.transform on exit
        itf.transform.push(Transform(ivec3(17,1,1)))
        placeBox(itf, Box(size=ivec3(3,3,3)), Block("acacia_planks"))

    # This common pattern can be simplified even further!
    with itf.pushTransform(Transform(ivec3(21,1,1))):
        placeBox(itf, Box(size=ivec3(3,3,3)), Block("dark_oak_planks"))

    # Several functions with a Transform parameter can also take a vector argument, which is
    # interpreted as a translation. This QoL feature removes the need to construct a Transform
    # when only a translation is needed. Interface.pushTransform() is one of these functions.
    with itf.pushTransform(ivec3(25,1,1)):
        placeBox(itf, Box(size=ivec3(3,3,3)), Block("crimson_planks"))

    # Build a staircase with various transformations
    def buildStaircase():
        for z in range(3):
            for y in range(z):
                placeBox(itf, Box(ivec3(0,y,z), ivec3(3,1,1)), Block("cobblestone"))
            placeBox(itf, Box(ivec3(0,z,z), ivec3(3,1,1)), Block("oak_stairs", facing="south"))

    for transform in [
        Transform(translation=ivec3(1,   1, 5  )),
        Transform(translation=ivec3(5+2, 1, 5  ), rotation=1),
        Transform(translation=ivec3(9,   1, 5+2), scale=ivec3(2,1,-1))
    ]:
        with itf.pushTransform(transform):
            buildStaircase()

    # When using a rotating or flipping transform, note that structures will extend in a different
    # direction: a box extending to positive X and Z will extend to negative X and positive Z when
    # rotated by 1. This is why there are correcting +2's in the offsets of the previous transforms.
    #
    # Use rotatedBoxTransform and/or scaledBoxTransform to transform to a rotated/scaled box
    # without needing to manually correct the offset. There's also a flippedBoxTransform.
    for transform in [
        rotatedBoxTransform(Box(ivec3(17,1,5), ivec3(3,3,3)), 1),
         scaledBoxTransform(Box(ivec3(21,1,5), ivec3(3,3,3)), ivec3(2,1,-1))
    ]:
        with itf.pushTransform(transform):
            buildStaircase()

    # Transforms can be muliplied like matrices.
    # From left to right, we stack "local coordinate systems". Note however, that the composite
    # transform is the equivalent of applying the subtransforms from right to left, not the other
    # way around.
    # Transform supports many more operations besides multiplication: refer to transform.py.
    t1 = Transform(ivec3(1,1,9))
    t2 = Transform(scale=ivec3(1,2,1))
    t3 = rotatedBoxTransform(Box(size=ivec3(3,3,3)), 1)
    transform = t1 @ t2 @ t3
    with itf.pushTransform(transform):
        placeBox(itf, Box(size=ivec3(1,1,3)), Block("sandstone"))

    # Place a block with NBT data
    with itf.pushTransform(ivec3(1,1,9)):
        itf.placeBlock(ivec3(1,0,1), Block("chest", facing="south", nbt='Items: [{Slot: 13, id: "apple", Count: 1}]'))

    # There are some NBT helpers available
    with itf.pushTransform(ivec3(5,1,9)):
        nbt = signNBT(line2="Hello, world!", color="blue")
        placeBox(itf, Box(ivec3(1,0,1), ivec3(1,2,1)), Block("stone"))
        itf.placeBlock(ivec3(1,1,2), Block("oak_wall_sign", facing="south", nbt=nbt))

    # It is possible to build a model in minecraft, scan it in, and then place it from code!
    testShape = models.testShape
    with itf.pushTransform(ivec3(9,1,9)):
        testShape.build(itf)
        testShape.build(itf, rotatedBoxTransform(Box(ivec3( 5,0,0), testShape.size), 1))
        testShape.build(itf, flippedBoxTransform(Box(ivec3(10,0,0), testShape.size), bvec3(0,0,1)))

    # Models can be built with substitutions, making it possible to create "shape" models while
    # applying their "texture" later. Namespaced id's are required for the keys.
    testShape.build(
        itf,
        ivec3(24,1,9),
        substitutions={
            "minecraft:red_concrete":    "red_wool",
            "minecraft:blue_concrete":   "blue_wool",
            "minecraft:lime_concrete":   "lime_wool",
            "minecraft:yellow_concrete": "yellow_wool",
            "minecraft:purpur_stairs":   ["stone_stairs", "andesite_stairs", "cobblestone_stairs"]
        }
    )


def main():
    # Get the build area
    buildArea = getBuildArea()
    buildRect = buildArea.toRect()

    # Check whether the build area is large enough
    if any(buildRect.size < EXAMPLE_STRUCTURE_SIZE):
        eprint(f"The build area rectangle is too small! Its size needs to be at least {vecString(EXAMPLE_STRUCTURE_SIZE)}")
        sys.exit(1)

    # Get a world slice and a heightmap
    worldSlice = getWorldSlice(buildRect)
    heightmap  = worldSlice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]

    # Create an Interface object with a transform that translates to the build rect
    itf = Interface(addY(buildRect.offset))

    # Place build area indicator
    meanHeight = int(np.mean(heightmap))
    placeRectOutline(itf, Rect(size=buildRect.size), meanHeight + 20, Block("red_concrete"))

    # Build the example structure in the center of the build area, at the mean height.
    rect   = centeredSubRect(Rect(size=buildRect.size), EXAMPLE_STRUCTURE_SIZE)
    height = int(np.mean(rectSlice(heightmap, rect)))
    with itf.pushTransform(addY(rect.offset, height)):
        buildExampleStructure(itf)

    # Flush block buffer
    itf.sendBufferedBlocks()


if __name__ == "__main__":
    main()
