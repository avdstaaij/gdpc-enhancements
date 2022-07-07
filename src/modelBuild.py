#!/usr/bin/env python3

# Builds the model read from standard in at the bottom left corner of the buildarea.
# Used for testing

import sys
import os
import argparse
from glm import ivec3 # Import needed for eval

from util.util import eprint

from mc.vector_util import vecString, Box
from mc.transform import Transform, rotatedBoxTransform
from mc.interface import getBuildArea, Interface
from mc.block import Block # Import needed for eval
from mc.model import Model

import models


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Builds a saved minecraft model in the build area"
    )
    parser.add_argument(
        "model", nargs="?", type=str,
        help = "The model to place (read from \"models.py\"). Ignored if --eval is used."
    )
    parser.add_argument(
        "-e", "--eval", action="store_true",
        help = "eval() a model string passed through standard in, instead of reading a model from the model file."
    )
    parser.add_argument(
        "-r", "--rotation", type=int, default=0,
        help = "Optional rotation to apply to the model. Must be one of {0,1,2,3}."
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    if args.eval:
        modelString = sys.stdin.read()
        model: Model = eval(modelString) # pylint: disable=eval-used
    elif args.model is not None:
        model: Model = getattr(models, args.model)
    else:
        eprint("Error: specify either a model name, or use --eval.\nUse --help for more info.")
        sys.exit(1)

    if args.rotation not in [0, 1, 2, 3]:
        eprint("Error: rotation must be one of {0,1,2,3}.")
        sys.exit(1)

    buildArea = getBuildArea()

    eprint(f"Building model at {vecString(buildArea.offset)}")

    itf = Interface(Transform(buildArea.offset), buffering=True)
    model.build(itf, rotatedBoxTransform(Box(size=model.size), args.rotation))


if __name__ == '__main__':
    main()
