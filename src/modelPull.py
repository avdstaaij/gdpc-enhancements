#!/usr/bin/env python3

import sys
from typing import List
import argparse
from glm import ivec3

from util.util import eprint
from mc.vectorUtil import Area, rotateSize3D, rotatedAreaTransform, areaBetween
from mc.interface import getBuildArea, getWorldSlice
from mc.block import Block
from mc.model import Model


def dumpModel(area: Area, rotation: int, filteredBlocks: List[str], filteredBlocksBecomeAir: bool):
    eprint(f"Model from {area}:\n")

    worldSlice = getWorldSlice(area.toRect())
    transform = rotatedAreaTransform(area, rotation)

    model = Model(rotateSize3D(area.size, rotation))

    for vec in area.inner:
        blockCompound = worldSlice.getBlockCompoundAt(vec.x, vec.y, vec.z)
        block = Block("minecraft:air") if blockCompound is None else Block.fromBlockCompound(blockCompound, rotation)
        if str(block.name) in filteredBlocks:
            if filteredBlocksBecomeAir:
                block = Block("minecraft:air")
            else:
                continue
        localVec = transform.invApply(vec)
        model.blocks[localVec.x][localVec.y][localVec.z] = block

    print(repr(model)) # TODO: Perhaps use pickle or json instead


def getArguments():
    parser = argparse.ArgumentParser(
        description="Scans in a model from an open Minecraft session and dumps it as a string"
    )
    parser.add_argument(
        "-a", "--area", type=str, required=False,
        help="The area from which the model is scanned, specified as [x1 y1 z1 x2 y2 z2] (inclusive). Example: \"0 64 0 20 84 20\". Defaults to the build area, which in turn defaults to [0 0 0 128 256 128]."
    )
    parser.add_argument(
        "-r", "--rotation", type=int, default=0,
        help="Rotation applied to the scanned model. Should be 0, 1, 2 or 3 (default 0). This is interpreted as the rotation with which the model area is viewed, so the /inverse/ of it is added to resulting model."
    )
    parser.add_argument(
        "--filter", type=str, required=False,
        help="List of blocks to exclude from the model. Air is always excluded, unless --include-air is used. Blocks should be specified with their namespaced ID, separated by commas, and with no additional whitespace. Example: \"minecraft:dirt,minecraft:grass_block\"."
    )
    parser.add_argument(
        "--include-air", action="store_true",
        help="Include air in the model. To include only specific types of air, use this in conjunction with --filter."
    )
    parser.add_argument(
        "--filtered-blocks-become-air", action="store_true",
        help="Replace filtered blocks with minecraft:air instead of None (even if minecraft:air is filtered out, which it is by default)"
    )
    return parser.parse_args()


def main():
    args = getArguments()

    if args.area is None:
        area = getBuildArea()
    else:
        coords = args.area.split(" ")
        if len(coords) != 6:
            eprint(f"Error: --area requires 6 coordinates, but {len(coords)} were given.")
            sys.exit(1)
        area = areaBetween(
            ivec3(int(coords[0]), int(coords[1]), int(coords[2])),
            ivec3(int(coords[3]), int(coords[4]), int(coords[5]))
        )

    rotation = args.rotation

    filteredBlocks = []
    if not args.include_air:
        filteredBlocks += ["minecraft:air","minecraft:cave_air","minecraft:void_air"]
    if args.filter is not None:
        filteredBlocks += args.filter.split(",")

    filteredBlocksBecomeAir = args.filtered_blocks_become_air

    dumpModel(area, rotation, filteredBlocks, filteredBlocksBecomeAir)


if __name__ == '__main__':
    main()
