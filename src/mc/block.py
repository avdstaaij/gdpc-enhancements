from typing import Any, Union, Optional, List
from copy import deepcopy
from dataclasses import dataclass
from glm import bvec2

from .util import transformXZaxisString, transformXZfacingString


@dataclass
class Block:
    name:      Union[str, List[str]] = "minecraft:stone"
    axis:      Optional[str]         = None
    facing:    Optional[str]         = None
    otherData: Optional[str]         = None
    nbt:       Optional[str]         = None # Excluding the outer braces
    needs_late_placement: bool       = False # Whether the block needs to be placed after its neighbors


    def transform(self, rotation: int = 0, flip: bvec2 = bvec2()):
        if not self.axis   is None: self.axis   = transformXZaxisString  (self.axis,   rotation)
        if not self.facing is None: self.facing = transformXZfacingString(self.facing, rotation, flip)


    def transformed(self, rotation: int = 0, flip: bvec2 = bvec2()):
        return Block(
            name      = self.name,
            axis      = None if self.axis   is None else transformXZaxisString  (self.axis,   rotation),
            facing    = None if self.facing is None else transformXZfacingString(self.facing, rotation, flip),
            otherData = self.otherData,
            nbt       = self.nbt,
            needs_late_placement = self.needs_late_placement
        )


    def blockStateString(self, rotation: int = 0, flip: bvec2 = bvec2()):
        """ Returns a string containing the block state of this block """

        if self.axis is None and self.facing is None and self.otherData is None:
            return ""

        dataItems = []
        if not self.axis       is None: dataItems.append("axis="   + transformXZaxisString  (self.axis,   rotation))
        if not self.facing     is None: dataItems.append("facing=" + transformXZfacingString(self.facing, rotation, flip))
        if not self.otherData  is None: dataItems.append(self.otherData)
        return "[" + ",".join(dataItems) + "]"


    def __str__(self):
        data_string = self.blockStateString() + (self.nbt if self.nbt else "")
        if isinstance(self.name, str):
            return "" if self.name == "" else self.name + data_string
        return ",".join([(name if name == "" else name + data_string) for name in self.name])


    def __repr__(self):
        # The default repr includes unnecessary default values, which make model dumps way larger
        # than they need to be.
        def optFieldStr(name: str, value: Optional[Any]):
            return ("" if value is None else f",{name}={repr(value)}")
        return (
            f"Block(\"{self.name}\"" +
            optFieldStr("axis",      self.axis)      +
            optFieldStr("facing",    self.facing)    +
            optFieldStr("otherData", self.otherData) +
            optFieldStr("nbt",       self.nbt)       +
            (",needs_late_placement=True" if self.needs_late_placement else "") +
            ")"
        )


    @staticmethod
    def fromBlockCompound(blockCompound, rotation: int = 0):
        """
        [rotation] specifies the rotation with which we are looking at the block.
        That is, its /inverse/ will be added to the returned Block.
        """
        # TODO: read in NBT data here, if we ever need that in a model
        block = Block(str(blockCompound["Name"]))
        if "Properties" in blockCompound:
            properties = blockCompound["Properties"]
            dataItems = []
            for key in properties:
                value = str(properties[key])
                if key in ["shape", "north", "east", "south", "west"]:
                    # This is a late property. We drop it, but set needs_late_placement to True
                    block.needs_late_placement = True
                elif key == "axis":
                    block.axis = value
                elif key == "facing":
                    block.facing = value
                else:
                    dataItems.append(str(key) + "=" + value)
            if dataItems:
                block.otherData = ",".join(dataItems)

        inverseRotation = (-rotation + 4) % 4
        block.transform(inverseRotation)

        return block
