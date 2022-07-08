""" Contains generated model code """

# This file contains generated model code. It's not intended to be readable.
# Code folding is recommended. (In VS Code: Ctrl-K Ctrl-0.)

__author__    = "Arthur van der Staaij"
__copyright__ = "Copyright 2022, Arthur van der Staaij"
__licence__   = "MIT"


from glm import ivec3

from gdpce.model import Model
from gdpce.block import Block


testShape=Model(
    size=ivec3( 4, 4, 4 ), blocks=[Block("minecraft:yellow_concrete"), Block("minecraft:blue_concrete"), Block("minecraft:blue_concrete"), Block("minecraft:blue_concrete"), Block("minecraft:lime_concrete"), None, None, None, Block("minecraft:lime_concrete"), None, None, None, Block("minecraft:lime_concrete"), None, None, None, Block("minecraft:red_concrete"), Block("minecraft:purpur_stairs",facing='west',otherState='half=bottom,waterlogged=false',needsLatePlacement=True), Block("minecraft:purpur_stairs",facing='west',otherState='half=bottom,waterlogged=false',needsLatePlacement=True), Block("minecraft:purpur_stairs",facing='south',otherState='half=bottom,waterlogged=false',needsLatePlacement=True), None, None, None, None, None, None, None, None, None, None, None, None, Block("minecraft:red_concrete"), Block("minecraft:purpur_stairs",facing='north',otherState='half=bottom,waterlogged=false',needsLatePlacement=True), None, None, None, None, None, None, None, None, None, None, None, None, None, None, Block("minecraft:red_concrete"), Block("minecraft:purpur_stairs",facing='east',otherState='half=bottom,waterlogged=false',needsLatePlacement=True), None, Block("minecraft:purpur_slab",otherState='waterlogged=false,type=bottom'), None, None, None, None, None, None, None, None, None, None, None, None])
