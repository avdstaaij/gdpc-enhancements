# This file contains models created using modelPull.py. It's not intended to be readable.
# Code folding is recommended. (In VS Code: Ctrl-K Ctrl-0.)

from glm import ivec3

from mc.model import Model
from mc.block import Block


testShape=Model(
    size=ivec3( 4, 3, 4 ), blocks=[[[Block("minecraft:yellow_concrete"), Block("minecraft:blue_concrete"), Block("minecraft:blue_concrete"), Block("minecraft:blue_concrete")], [Block("minecraft:lime_concrete"), None, None, None], [Block("minecraft:lime_concrete"), None, None, None]], [[Block("minecraft:red_concrete"), Block("minecraft:purpur_stairs",facing='north',otherData='half=bottom,waterlogged=false',needs_late_placement=True), Block("minecraft:purpur_stairs",facing='west',otherData='half=bottom,waterlogged=false',needs_late_placement=True), Block("minecraft:purpur_stairs",facing='south',otherData='half=bottom,waterlogged=false',needs_late_placement=True)], [None, None, None, None], [None, None, None, None]], [[Block("minecraft:red_concrete"), Block("minecraft:purpur_stairs",facing='north',otherData='half=bottom,waterlogged=false',needs_late_placement=True), None, None], [None, None, None, None], [None, None, None, None]], [[Block("minecraft:red_concrete"), Block("minecraft:purpur_stairs",facing='east',otherData='half=bottom,waterlogged=false',needs_late_placement=True), None, Block("minecraft:purpur_slab",otherData='waterlogged=false,type=bottom')], [None, None, None, None], [None, None, None, None]]])
