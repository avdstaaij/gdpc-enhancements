from dataclasses import dataclass
from glm import ivec3, bvec2


def to_namespaced_id(block_name: str):
    if ":" in block_name:
        return block_name
    return f"minecraft:{block_name}" # This works so long as we're not using mods


def axisVectorToString(vec: ivec3):
    m = {
        (1, 0, 0): "x",
        (0, 1, 0): "y",
        (0, 0, 1): "z",
    }
    v = (
        vec.x != 0,
        vec.y != 0,
        vec.z != 0,
    )
    try:
        return m[v]
    except KeyError as e:
        raise ValueError("axisVectorToString: exactly one vector component must be non-zero") from e


def axisStringToVector(axis: str):
    m = {
        "x": ivec3(1,0,0),
        "y": ivec3(0,1,0),
        "z": ivec3(0,0,1),
    }
    return m[axis]


def facingVectorToString(vec: ivec3):
    m = {
        ( 0, 0,-1): "north",
        ( 0, 0, 1): "south",
        ( 0,-1, 0): "down",
        ( 0, 1, 0): "up",
        (-1, 0, 0): "west",
        ( 1, 0, 0): "east",
    }
    v = (
        -1 if vec[0] < 0 else 1 if vec[0] > 0 else 0,
        -1 if vec[1] < 0 else 1 if vec[1] > 0 else 0,
        -1 if vec[2] < 0 else 1 if vec[2] > 0 else 0,
    )
    try:
        return m[v]
    except KeyError as e:
        raise ValueError("facingVectorToString: exactly one vector component must be non-zero") from e


def facingStringToVector(direction: str):
    m = {
        "north": ivec3( 0, 0,-1),
        "south": ivec3( 0, 0, 1),
        "down":  ivec3( 0,-1, 0),
        "up":    ivec3( 0, 1, 0),
        "west":  ivec3(-1, 0, 0),
        "east":  ivec3( 1, 0, 0),
    }
    return m[direction]


def rotateXZaxisString(axis: str, rotation: int):
    strings = ["x", "z"]
    try:
        return strings[(strings.index(axis) + rotation) % 2]
    except ValueError:
        return axis # y


def transformXZaxisString(axis: str, rotation: int = 0):
    # Flipping is a no-op for axis strings
    return rotateXZaxisString(axis, rotation)


def rotateXZfacingString(facing: str, rotation: int):
    strings = ["north", "east", "south", "west"]
    try:
        return strings[(strings.index(facing) + rotation) % 4]
    except ValueError:
        return facing # up, down


def flipXZfacingString(facing: str, flip: bvec2):
    if flip.x:
        if facing == "east": return "west"
        if facing == "west": return "east"
    if flip.y: # "z"
        if facing == "north": return "south"
        if facing == "south": return "north"
    return facing


def transformXZfacingString(facing: str, rotation: int, flip: bvec2 = bvec2()):
    return rotateXZfacingString(flipXZfacingString(facing, flip), rotation)
