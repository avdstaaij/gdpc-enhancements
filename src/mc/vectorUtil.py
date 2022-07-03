import random
from typing import Any, List, Union, overload
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
import glm
from glm import ivec2, ivec3, vec2, vec3
import math


UP      = ivec3( 0, 1, 0)
DOWN    = ivec3( 0,-1, 0)
LEFT    = ivec3( 1, 0, 0)
RIGHT   = ivec3(-1, 0, 0)
FORWARD = ivec3( 0, 0, 1)
BACK    = ivec3( 0, 0,-1)
XYZ     = ivec3( 1, 1, 1)
XZ      = ivec3( 1, 0, 1)


def dropY(vec: ivec3) -> ivec2:
    return vec.xz


def addY(vec: ivec2, y: int = 0):
    return ivec3(vec.x, y, vec.y)


def setY(vec: ivec3, y: int = 0):
    return ivec3(vec.x, y, vec.z)


@overload
def perpendicular(vec: vec2) -> vec2:
    ...
@overload
def perpendicular(vec: ivec2) -> ivec2:
    ...
def perpendicular(vec: Union[vec2, ivec2]):
    """ Returns a vector perpendicular to [vec] that points to the right of [vec], and has the same
        length as [vec]. """
    if isinstance(vec,  vec2): return  vec2(vec.y, -vec.x)
    if isinstance(vec, ivec2): return ivec2(vec.y, -vec.x)
    raise ValueError()


def rotateXZ(vec: ivec3, rotation: int):
    # rotation %= 4
    if rotation == 0:
        return vec
    if rotation == 1:
        return ivec3(-vec.z, vec.y, vec.x)
    if rotation == 2:
        return ivec3(-vec.x, vec.y, -vec.z)
    if rotation == 3:
        return ivec3(vec.z, vec.y, -vec.x)
    raise ValueError("rotateXZ: rotation must be in {0,1,2,3}")


def to_axis_vector(vec: ivec2):
    if abs(vec.x) > abs(vec.y):
        if vec.x < 0:
            return ivec2(-1, 0)
        else:
            return ivec2(1, 0)
    else:
        if vec.y < 0:
            return ivec2(0, -1)
        else:
            return ivec2(0, 1)


def direction_to_rotation(direction: ivec2):
    """ Returns the rotation that rotates (0,-1) closest to [direction]"""
    vec = to_axis_vector(direction)
    if vec.y < 0: return 0
    if vec.x > 0: return 1
    if vec.y > 0: return 2
    if vec.x < 0: return 3
    raise ValueError()


# For some reason, glm's length, length2, distance, distance2 and l1Norm refuse to work with integer
# vectors. We provide some wrappers.

def length(vec: Union[ivec2, ivec3]):
    if isinstance(vec, ivec2): return glm.length(vec2(vec))
    if isinstance(vec, ivec3): return glm.length(vec3(vec))
    raise ValueError()

def length2(vec: Union[ivec2, ivec3]):
    if isinstance(vec, ivec2): return int(glm.length2(vec2(vec)))
    if isinstance(vec, ivec3): return int(glm.length2(vec3(vec)))
    raise ValueError()

def distance(vecA: Union[ivec2, ivec3], vecB: Union[ivec2, ivec3]):
    if isinstance(vecA, ivec2) and isinstance(vecB, ivec2): return glm.distance(vec2(vecA), vec2(vecB))
    if isinstance(vecA, ivec3) and isinstance(vecB, ivec2): return glm.distance(vec3(vecA), vec3(vecB))
    raise ValueError()

def distance2(vecA: Union[ivec2, ivec3], vecB: Union[ivec2, ivec3]):
    if isinstance(vecA, ivec2) and isinstance(vecB, ivec2): return int(glm.distance2(vec2(vecA), vec2(vecB)))
    if isinstance(vecA, ivec3) and isinstance(vecB, ivec2): return int(glm.distance2(vec3(vecA), vec3(vecB)))
    raise ValueError()

def l1Norm(vec: Union[ivec2, ivec3]):
    if isinstance(vec, ivec2): return abs(vec.x) + abs(vec.y)
    if isinstance(vec, ivec3): return abs(vec.x) + abs(vec.y) + abs(vec.z)
    raise ValueError()

def l1Distance(vecA: Union[ivec2, ivec3], vecB: Union[ivec2, ivec3]):
    return l1Norm(vecA - vecB)

def vecString(vec: Union[ivec2, ivec3]):
    """ The default __str__ is not very nice to read """
    if isinstance(vec, ivec2):
        return f"({vec.x}, {vec.y})"
    if isinstance(vec, ivec3):
        return f"({vec.x}, {vec.y}, {vec.z})"
    return ""


@dataclass
class Transform:
    """ Represents a transformation of space.
        When applied to a vector, [scaling] and [rotation] are applied first (they are independent),
        and [translation] is applied second.
        Note that only integer scaling is supported. This has some notable side-effects, such as
        t1.compose(t2.inverted()) not always being equivalent to t1.composeInv(t2).
        """

    translation: ivec3 = field(default_factory=ivec3)
    rotation:    int   = 0 # XZ-rotation; 0, 1, 2 or 3
    scale:       ivec3 = field(default_factory=lambda: ivec3(1,1,1)) # Only integer scaling!

    def apply(self, vec: ivec3):
        """ Applies this transform to [vec]
            Equivalent to [self] * [vec] """
        return rotateXZ(vec * self.scale, self.rotation) + self.translation

    def invApply(self, vec: ivec3):
        """ Applies the inverse of this transform to [vec]
            Faster version of ~[self] * [vec] that is safer when using non-unit scalings """
        return rotateXZ(vec - self.translation, (-self.rotation + 4) % 4) / self.scale

    def compose(self, other: 'Transform'):
        """ Returns a transform that applies [self] after [other]
            Equivalent to [self] @ [other] """
        return Transform(
            translation = self.apply(other.translation),
            rotation    = (self.rotation + other.rotation) % 4,
            scale       = self.scale * other.scale
        )

    def invCompose(self, other: 'Transform'):
        """ Returns a transform that applies [self]^-1 after [other]
            Faster version of ~[self] @ other that is safer when using non-unit scalings """
        return Transform(
            translation = self.invApply(other.translation),
            rotation    = (other.rotation - self.rotation + 4) % 4,
            scale       = other.scale / self.scale # Safer than (1/self.scaling) * other.scaling
        )

    def composeInv(self, other: 'Transform'):
        """ Returns a transform that applies [self] after [other]^-1
            Faster version of [self] @ ~[other] that is safer when using non-unit scalings """
        scaling  = self.scale / other.scale
        rotation = (self.rotation - other.rotation + 4) % 4
        return Transform(
            translation = self.translation - rotateXZ(other.translation * scaling, rotation),
            rotation    = rotation,
            scale       = scaling
        )

    def push(self, other: 'Transform'):
        """ Adds the effect of [other] to this transform
            Equivalent to [self] @= [other] """
        self.translation += rotateXZ(other.translation * self.scale, self.rotation)
        self.rotation     = (self.rotation + other.rotation) % 4
        self.scale       *= other.scale

    def pop(self, other: 'Transform'):
        """ The inverse of push. Removes the effect of [other] from this transform
            Faster version of [self] @= ~[other] that is safer when using non-unit scalings """
        self.scale       /= other.scale
        self.rotation     = (self.rotation - other.rotation + 4) % 4
        self.translation -= rotateXZ(other.translation * self.scale, self.rotation)

    def inverted(self):
        """ Equivalent to ~[self].
            Note that non-unit scalings cannot be inverted: any fractional part is dropped. """
        scaling  = 1 / self.scale
        rotation = (-self.rotation + 4) % 4
        return Transform(
            translation = rotateXZ(self.translation, rotation) * scaling,
            rotation    = rotation,
            scale       = scaling
        )

    def invert(self):
        """ Faster equivalent of [self] = ~[self].
            Note that non-unit scalings cannot be inverted: any fractional part is dropped. """
        self.scale       = 1 / self.scale
        self.rotation    = (-self.rotation + 4) % 4
        self.translation = rotateXZ(self.translation, self.rotation) * self.scale

    def __matmul__(self, other: 'Transform') -> 'Transform':
        return self.compose(other)

    def __mul__(self, vec: ivec3) -> ivec3:
        return self.apply(vec)

    def __imatmul__(self, other: 'Transform'):
        self.push(other)
        return self

    def __invert__(self):
        return self.inverted()


@dataclass
class Rect:
    offset: ivec2 = field(default_factory=ivec2)
    size:   ivec2 = field(default_factory=ivec2)

    @property
    def begin(self):
        return self.offset

    @begin.setter
    def begin(self, value: ivec2):
        self.offset = value

    @property
    def end(self):
        return self.begin + self.size

    @end.setter
    def end(self, value: ivec2):
        self.size = value - self.begin

    @property
    def middle(self):
        return self.begin + self.size / 2

    @property
    def inner(self):
        return (
            ivec2(x, y)
            for x in range(self.begin.x, self.end.x)
            for y in range(self.begin.y, self.end.y)
        )

    @property
    def area(self):
        return self.size.x*self.size.y

    @property
    def corners(self):
        return [self.offset, self.offset + ivec2(self.size.x, 0),  self.end, self.offset + ivec2(0, self.size.y)]

    def contains(self, vec: ivec2):
        return (
            self.begin.x <= vec.x < self.end.x and
            self.begin.y <= vec.y < self.end.y
        )

    def collides(self, other: 'Rect'):
        return (
            self.begin.x <= other.end  .x and
            self.end  .x >= other.begin.x and
            self.begin.y <= other.end  .y and
            self.end  .y >= other.begin.y
        )

    def distanceToVecSquared(self, other: ivec2):
        dx = max(self.begin.x - other.x, 0, other.x - (self.end.x - 1))
        dy = max(self.begin.y - other.y, 0, other.y - (self.end.y - 1))
        return dx*dx + dy*dy

    def translated(self, translation: Union[ivec2, int]):
        return Rect(self.offset + translation, self.size)

    def dilate(self, dilation: int = 1):
        self.offset -= dilation
        self.size   += dilation*2

    def dilated(self, dilation: int = 1):
        return Rect(self.offset - dilation, self.size + dilation*2)

    def erode(self, erosion: int = 1):
        self.dilate(-erosion)

    def eroded(self, erosion: int = 1):
        return self.dilated(-erosion)

    def toArea(self, offsetY = 0, sizeY = 0):
        return Area(addY(self.offset, offsetY), addY(self.size, sizeY))

    def __str__(self):
        return f"{vecString(self.begin)} - {vecString(self.end)}"


@dataclass
class Area:
    offset: ivec3 = field(default_factory=ivec3)
    size:   ivec3 = field(default_factory=ivec3)

    @property
    def begin(self):
        return self.offset

    @begin.setter
    def begin(self, value: ivec2):
        self.offset = value

    @property
    def end(self):
        return self.begin + self.size

    @end.setter
    def end(self, value: ivec2):
        self.size = value - self.begin

    @property
    def middle(self):
        return self.begin + self.size / 2

    @property
    def inner(self):
        return (
            ivec3(x, y, z)
            for x in range(self.begin.x, self.end.x)
            for y in range(self.begin.y, self.end.y)
            for z in range(self.begin.z, self.end.z)
        )

    def contains(self, vec: ivec3):
        return (
            self.begin.x <= vec.x < self.end.x and
            self.begin.y <= vec.y < self.end.y and
            self.begin.z <= vec.z < self.end.z
        )

    def collides(self, other: 'Area'):
        return (
            self.begin.x <= other.end  .x and
            self.end  .x >= other.begin.x and
            self.begin.y <= other.end  .y and
            self.end  .y >= other.begin.y and
            self.begin.z <= other.end  .z and
            self.end  .z >= other.begin.z
        )

    def translated(self, translation: Union[ivec3, int]):
        return Area(self.offset + translation, self.size)

    def dilate(self, dilation: int = 1):
        self.offset -= dilation
        self.size   += dilation*2

    def dilated(self, dilation: int = 1):
        return Rect(self.offset - dilation, self.size + dilation*2)

    def erode(self, erosion: int = 1):
        self.dilate(-erosion)

    def eroded(self, erosion: int = 1):
        return self.dilated(-erosion)

    def toRect(self):
        return Rect(dropY(self.offset), dropY(self.size))

    def __str__(self):
        return f"{vecString(self.begin)} - {vecString(self.end)}"


def rectBetween(cornerA: ivec2, cornerB: ivec2):
    """ Returns the Rect between [cornerA] and [cornerB], which may be any opposing corners """
    first = ivec2(
        cornerA.x if cornerA.x <= cornerB.x else cornerB.x,
        cornerA.y if cornerA.y <= cornerB.y else cornerB.y,
    )
    last = ivec2(
        cornerA.x if cornerA.x > cornerB.x else cornerB.x,
        cornerA.y if cornerA.y > cornerB.y else cornerB.y,
    )
    return Rect(first, last - first + 1)


def areaBetween(cornerA: ivec3, cornerB: ivec3):
    """ Returns the Area between [cornerA] and [cornerB], which may be any opposing corners """
    first = ivec3(
        cornerA.x if cornerA.x <= cornerB.x else cornerB.x,
        cornerA.y if cornerA.y <= cornerB.y else cornerB.y,
        cornerA.z if cornerA.z <= cornerB.z else cornerB.z,
    )
    last = ivec3(
        cornerA.x if cornerA.x > cornerB.x else cornerB.x,
        cornerA.y if cornerA.y > cornerB.y else cornerB.y,
        cornerA.z if cornerA.z > cornerB.z else cornerB.z,
    )
    return Area(first, last - first + 1)


def centeredSubRectOffset(rect: Rect, size: ivec2):
    """ Returns an offset such that the rect <offset, [size]> is centered in [rect].
        If [size] is larger than [rect].size in an axis, [rect] will instead be centered in
        <offset, [size]> in that axis. """
    difference = rect.size - size
    return rect.offset + difference/2


def centeredSubAreaOffset(area: Area, size: ivec2):
    """ Returns an offset such that the area <offset, [size]> is centered in [area].
        If [size] is larger than [area].size in an axis, [area] will instead be centered in
        <offset, [size]> in that axis. """
    difference = area.size - size
    return area.offset + difference/2


def centeredSubRect(rect: Rect, size: ivec2):
    """ Returns a rect of size [size] that is centered in [rect].
        If [size] is larger than [rect].size in an axis, [rect] will instead be centered in the
        returned rect in that axis. """
    return Rect(centeredSubRectOffset(rect, size), size)


def centeredSubArea(area: Rect, size: ivec2):
    """ Returns a area of size [size] that is centered in [area].
        If [size] is larger than [area].size in an axis, [area] will instead be centered in the
        returned area in that axis. """
    return Area(centeredSubAreaOffset(area, size), size)


def rotateSize2D(size: ivec2, rotation: int):
    """ Returns the size of a rect of size [size] that has been rotated under [rotation] """
    return ivec2(size.y, size.x) if rotation in [1, 3] else size


def rotateSize3D(size: ivec3, rotation: int):
    """ Returns the size of an area of size [size] that has been rotated under [rotation] """
    return addY(rotateSize2D(dropY(size), rotation), size.y)


def rotatedAreaTransform(area: Area, rotation: int):
    """
    Returns a transform that maps the area <(0,0,0) - rotateSize3D([area].size, [rotation])>
    to [area], under [rotation].
    Note that rotateSize3D([area].size, [rotation]) == [area].size iff [area] is square in the
    XZ-plane or [rotation] is 0 or 2
    """
    return Transform(
        translation = area.offset + ivec3(
            area.size.x - 1 if rotation in [1, 2] else 0,
            0,
            area.size.z - 1 if rotation in [2, 3] else 0,
        ),
        rotation = rotation
    )


# TODO: flippedAreaTransform, or even just areaTransform?
def flipXAreaTransform(area: Area):
    return Transform(
        translation=ivec3((area.size.x-1), 0, 0),
        scale=ivec3(-1, 1, 1)
    )


def flipZAreaTransform(area: Area):
    return Transform(
        translation=ivec3(0, 0, (area.size.z-1)),
        scale=ivec3(1, 1, -1)
    )


def line_to_pixels_numpy(line: Range[ivec2], width: int = 1):
    delta = np.array(line.max - line.min)
    max_delta = int(max(abs(delta)))
    if max_delta == 0:
        return np.array([])
    pixels = delta[np.newaxis,:] * np.arange(max_delta + 1)[:,np.newaxis] / max_delta + np.array(line.min)
    pixels = np.rint(pixels).astype(np.signedinteger)

    if width > 1:
        min_pixel = np.array([min(line.max.x, line.min.x), min(line.max.y, line.min.y)])

        # convert pixel list to np array
        array = np.zeros((max_delta + width*2, max_delta + width*2), dtype=int)
        array[tuple(np.transpose(pixels - min_pixel + width))] = 1

        # dilate pixel array (make it THICC)
        if width > 1:
            array = ndimage.binary_dilation(array, iterations = width - 1)

        # rebuild pixel list from array
        pixels = np.argwhere(array) + min_pixel - width

    return pixels

def line_to_voxels_numpy(line: Range[ivec3], width: int = 1):
    delta = np.array(line.max - line.min)
    max_delta = int(max(abs(delta)))
    if max_delta == 0:
        return np.array([])
    voxels = delta[np.newaxis,:] * np.arange(max_delta + 1)[:,np.newaxis] / max_delta + np.array(line.min)
    voxels = np.rint(voxels).astype(np.signedinteger)

    if width > 1:
        min_voxel = np.array([min(line.max.x, line.min.x), min(line.max.y, line.min.y), min(line.max.z, line.min.z)])

        # convert pixel list to np array
        array_width = max_delta + width*2
        array = np.zeros((array_width, array_width, array_width), dtype=int)
        array[tuple(np.transpose(voxels - min_voxel + width))] = 1

        # dilate pixel array (make it THICC)
        if width > 1:
            array = ndimage.binary_dilation(array, iterations = width - 1)

        # rebuild pixel list from array
        voxels = np.argwhere(array) + min_voxel - width

    return voxels


def line_to_pixels(line: Range[ivec2], width: int = 1):
    pixel_array = line_to_pixels_numpy(line, width)
    return [ivec2(pixel[0], pixel[1]) for pixel in pixel_array]


def line_to_voxels(line: Range[ivec3], width: int = 1) -> List[ivec3]:
    voxel_array = line_to_voxels_numpy(line, width)
    return [ivec3(pixel[0], pixel[1], pixel[2]) for pixel in voxel_array]

def rect_slice(array: np.ndarray, rect: Rect):
    return array[rect.begin.x:rect.end.x, rect.begin.y:rect.end.y]


def set_rect_slice(array: np.ndarray, rect: Rect, value: Any):
    array[rect.begin.x:rect.end.x, rect.begin.y:rect.end.y] = value


def find_point_closest_to_rect(rect: Rect, point_array: np.ndarray):
    """ Returns the point from [point_array] closest to [rect] and the point of [rect] it is
        closest to, according to the L1 distance """
    assert point_array.shape[1] == 2, "point_array should be a list of 2-dimensional points"
    # This can surely be simplified
    d1 = np.array([rect.begin.x, rect.begin.y]) - point_array
    d2 = point_array - np.array([rect.end.x - 1, rect.end.y - 1])
    d = np.maximum(0, np.maximum(d1, d2))
    distances = np.sum(d, axis=1)
    index = np.argmin(distances)
    closest_road_point = point_array[index]
    side_sign = 2 * (closest_road_point >= [rect.end.x, rect.end.y]) - 1
    rect_point = closest_road_point - side_sign * d[index]
    return ivec2(closest_road_point[0], closest_road_point[1]), ivec2(rect_point[0], rect_point[1])


def is_rect_horizontal_to_points(rect: Rect, point_array: np.ndarray):
    """ Returns whether [rect] is oriented horizontally with respect to the point from [point_array]
        that is closest to it. Always returns True for square rects. """
    if rect.size.x == rect.size.y: return True
    closest_point, rect_point = find_point_closest_to_rect(rect, point_array)
    direction = to_axis_vector(closest_point - rect_point)
    return bool(direction[0]) ^ bool(rect.size.x > rect.size.y)


def is_rect_vertical_to_points(rect: Rect, point_array: np.ndarray):
    """ Returns whether [rect] is oriented vertically with respect to the point from [point_array]
        that is closest to it. Always returns True for square rects. """
    if rect.size.x == rect.size.y: return True
    return not is_rect_horizontal_to_points(rect, point_array)


def neighbors(point: ivec2, bounding_rect: Rect, diagonal: bool = False, stride: int = 1):
    """ Returns the neighbors of [point] within [bounding_rect]. Useful for custom astar calls. """

    # The common case of [diagonal]=False might be sped up by computing these booleans only for the
    # diagonal part.

    end = bounding_rect.end

    left   = point.x - stride >= bounding_rect.offset.x
    bottom = point.y - stride >= bounding_rect.offset.y
    right  = point.x + stride <  end.x
    top    = point.y + stride <  end.y

    if left:   yield ivec2(point.x - stride, point.y         )
    if bottom: yield ivec2(point.x         , point.y - stride)
    if right:  yield ivec2(point.x + stride, point.y         )
    if top:    yield ivec2(point.x         , point.y + stride)

    if not diagonal:
        return

    if left  and bottom: yield ivec2(point.x - stride, point.y - stride)
    if left  and top:    yield ivec2(point.x - stride, point.y + stride)
    if right and bottom: yield ivec2(point.x + stride, point.y - stride)
    if right and top:    yield ivec2(point.x + stride, point.y + stride)
