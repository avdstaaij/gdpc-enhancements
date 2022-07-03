from typing import List
import random
from glm import ivec2

from mc.vectorUtil import Rect, Transform, addY, centeredSubRectOffset, dropY, rotateSize2D, rotatedAreaTransform


def centerTransform(rect: Rect, structure_size: ivec2, y: int = 0):
    """ Returns a transform that translates a structure of size [structure_size] such that it is
        centered in [rect]. """
    return Transform(addY(centeredSubRectOffset(rect, structure_size), y))


def rotationTransform(rect: Rect, rotation: int, y: int = 0):
    """ Returns (transform, size) such that [transform] maps the rect <(0,0) - [size]> to [rect],
        under [rotation]. """
    return rotatedAreaTransform(rect.toArea(y), rotation), rotateSize2D(rect.size, rotation)


def fittingRotationTransform(rect: Rect, structure_size: ivec2, y: int = 0):
    """ Returns (transform, size) such that [transform] maps the rect <(0,0) - [size]> to [rect] and
        [structure_size] fits in [size]. If there are multiple rotations that achieve this, one is
        chosen at random. If there is no rotation that achieves this, a rotation is selected fully
        randomly. """
    rotation_candidates: List[int] = []
    if structure_size.x <= rect.size.x and structure_size.y <= rect.size.y:
        rotation_candidates += [0, 2]
    if structure_size.x <= rect.size.y and structure_size.y <= rect.size.x:
        rotation_candidates += [1, 3]
    if not rotation_candidates:  # Cannot fit
        rotation_candidates = [0, 1, 2, 3]
    rotation = random.choice(rotation_candidates)
    return rotationTransform(rect, rotation, y)


def horizontalTransform(rect: Rect, y: int = 0):
    """ Returns (transform, size) such that [transform] maps the rect <(0,0) - [size]> to [rect] and
        size.x >= size.y. If there are multiple rotations that achieve this, one is chosen a
        random. """
    if rect.size.x > rect.size.y:
        rotation_candidates = [0, 2]
    elif rect.size.x < rect.size.y:
        rotation_candidates = [1, 3]
    else:
        rotation_candidates = [0, 1, 2, 3]
    rotation = random.choice(rotation_candidates)
    return rotationTransform(rect, rotation, y)


def bottomToDoorTransform(rect: Rect, doorPosition: ivec2, y: int = 0):
    """ Returns (transform, size, doorX) such that [transform] maps the rect <(0,0) - [size]> to
        [rect] under the rotation that moves ([doorX], 0) to [doorPosition].
        Note that this means that [doorPosition] should be relative to [rect]. """
    if doorPosition.y == 0:
        rotation = 0
    elif doorPosition.x == rect.size.x - 1:
        rotation = 1
    elif doorPosition.y == rect.size.y - 1:
        rotation = 2
    elif doorPosition.x == 0:
        rotation = 3
    else:
        raise ValueError("The door is not at the edge of the rect")

    transform, size = rotationTransform(rect, rotation, y)

    doorPosition = dropY(transform.invApply(addY(doorPosition + rect.offset)))
    assert doorPosition.y == 0, "Wait, what?"

    return transform, size, doorPosition.x


# The xCenterTransform functions below are like the corresponding xTransform function, but they only
# return a Transform (no size) that additionally translates a structure of size [structure_size]
# such that it is centered in the transformed [rect].


def rotationCenterTransform(rect: Rect, rotation: int, structure_size: ivec2, y: int = 0):
    t1, size = rotationTransform(rect, rotation, y)
    t2 = centerTransform(Rect(size=size), structure_size)
    return t1 @ t2


def fittingRotationCenterTransform(rect: Rect, structure_size: ivec2, y: int = 0):
    t1, size = fittingRotationTransform(rect, structure_size, y)
    t2 = centerTransform(Rect(size=size), structure_size)
    return t1 @ t2


def horizontalCenterTransform(rect: Rect, structure_size: ivec2, y: int = 0):
    t1, size = horizontalTransform(rect, y)
    t2 = centerTransform(Rect(size=size), structure_size)
    return t1 @ t2
