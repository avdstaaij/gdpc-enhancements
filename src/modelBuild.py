#!/usr/bin/env python3

""" Executable script that builds saved models. Use --help for more information. """

__author__    = "Arthur van der Staaij"
__copyright__ = "Copyright 2022, Arthur van der Staaij"
__licence__   = "MIT"


from importlib import import_module
import sys
import argparse
from glm import ivec3 # Import needed for eval

from util.util import eprint

from gdpce.vector_util import vecString, Box
from gdpce.transform import Transform, rotatedBoxTransform
from gdpce.interface import getBuildArea, Interface
from gdpce.block import Block # Import needed for eval
from gdpce.model import Model


DEFAULT_MODEL_MODULE = "models"


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Builds a saved Minecraft model in the build area. Useful for development."
    )
    parser.add_argument(
        "model", nargs="?", type=str,
        help = f"The model to place, specified as \"[module].[model]\". Ignored if --eval is used. Example: \"path.to.models.house\". If the string contains no \".\", the prefix \"{DEFAULT_MODEL_MODULE}.\" is added (i.e. the default model file is \"models.py\")."
    )
    parser.add_argument(
        "-e", "--eval", action="store_true",
        help = "eval() a model string passed through standard in, instead of reading a model from a model file."
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
        moduleName, dot, modelName = args.model.rpartition(".")
        if dot == "": moduleName = DEFAULT_MODEL_MODULE
        model = getattr(import_module(moduleName), modelName)
    else:
        eprint("Error: specify either a model path, or use --eval.\nUse --help for more info.")
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
