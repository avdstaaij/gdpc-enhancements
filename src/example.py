#!/usr/bin/env python3

import sys

import numpy as np
from glm import ivec2, ivec3, bvec3

from mc.vector_util import addY, vecString, Rect, Box, centeredSubRectOffset, rectSlice
from mc.transform import Transform, rotatedBoxTransform, scaledBoxTransform, flippedBoxTransform
from mc.nbt_util import signNBT
from mc.block import Block
from mc.interface import Interface, getBuildArea, getWorldSlice
from mc.geometry import placeBox, placeRectOutline, placeCheckeredBox

from util.util import eprint

import models


EXAMPLE_STRUCTURE_SIZE = ivec2(43, 10)


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

    # Build a staircase with various transformations.
    def buildStaircase():
        for z in range(3):
            for y in range(z):
                placeBox(itf, Box(ivec3(0,y,z), ivec3(3,1,1)), Block("cobblestone"))
            placeBox(itf, Box(ivec3(0,z,z), ivec3(3,1,1)), Block("oak_stairs", facing="south"))

    for transform in [
        Transform(translation=ivec3(17,   1, 1  )),
        Transform(translation=ivec3(21+2, 1, 1  ), rotation=1),
        Transform(translation=ivec3(25,   1, 1+2), scale=ivec3(2,1,-1))
    ]:
        itf.transform.push(transform)
        buildStaircase()
        itf.transform.pop(transform)

    # When using a rotating or flipping transform, note that structures will extend in a different
    # direction: a box extending to positive X and Z will extend to negative X and positive Z when
    # rotated by 1. This is why there are correcting +2's in the offsets of the previous transforms.
    #
    # Use rotatedBoxTransform and/or scaledBoxTransform to transform to a rotated/scaled box
    # without needing to manually correct the offset. There's also a flippedBoxTransform.
    for transform in [
        rotatedBoxTransform(Box(ivec3(32,1,1), ivec3(3,3,3)), 1),
         scaledBoxTransform(Box(ivec3(36,1,1), ivec3(3,3,3)), ivec3(2,1,-1))
    ]:
        itf.transform.push(transform)
        buildStaircase()
        itf.transform.pop(transform)

    # Transforms can be muliplied like matrices.
    # From left to right, we stack "local coordinate systems". Note however, that the composite
    # transform is the equivalent of applying the subtransforms from right to left, not the other
    # way around.
    # Transform supports many more operations besides multiplication: refer to transform.py.
    t1 = Transform(ivec3(1,1,5))
    t2 = Transform(scale=ivec3(1,2,1))
    t3 = rotatedBoxTransform(Box(size=ivec3(3,3,3)), 1)
    transform = t1 @ t2 @ t3
    itf.transform.push(transform)
    placeBox(itf, Box(size=ivec3(3,1,3)), Block("sandstone"))
    itf.transform.pop(transform)

    # Place a block with NBT data
    itf.placeBlock(ivec3(6,1,6), Block("chest", facing="south", nbt='Items: [{Slot: 13, id: "apple", Count: 1}]'))

    # There are some helpers available
    placeBox(itf, Box(ivec3(10,1,6), ivec3(1,2,1)), Block("stone"))
    itf.placeBlock(ivec3(10,2,7), Block("oak_wall_sign", facing="south", nbt=signNBT(line2="Hello, world!", color="blue")))

    # It is possible to build a model in minecraft, scan it in, and then place it from code!
    testShape = models.testShape
    testShape.build(itf, Transform(ivec3(13,1,5)))
    testShape.build(itf, rotatedBoxTransform(Box(ivec3(18,1,5), testShape.size), 1))
    testShape.build(itf, flippedBoxTransform(Box(ivec3(23,1,5), testShape.size), bvec3(0,0,1)))
    testShape.build(itf, flippedBoxTransform(Box(ivec3(28,1,5), testShape.size), bvec3(0,1,0)))

    # Models can be built with substitutions, making it possible to create "shape" models while
    # applying their "texture" later. Namespaced id's are required for the keys.
    testShape.build(
        itf,
        Transform(ivec3(33,1,5)),
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

    # Create an Interface object
    itf = Interface()

    # Place build area indicator
    meanHeight = int(np.mean(heightmap))
    placeRectOutline(itf, buildRect, meanHeight + 20, Block("red_concrete"))

    # Build the example structure in the center of the build area, at the mean height.
    offset = centeredSubRectOffset(buildRect, EXAMPLE_STRUCTURE_SIZE)
    height = int(np.mean(rectSlice(heightmap, Rect(size=EXAMPLE_STRUCTURE_SIZE))))
    transform = Transform(addY(offset, height))
    itf.transform.push(transform)
    buildExampleStructure(itf)
    itf.transform.pop(transform)

    # Flush block buffer
    itf.sendBufferedBlocks()


if __name__ == "__main__":
    main()
