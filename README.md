# GDPC enhancements

This repository contains various enhancements to
[GDPC](https://github.com/nilsgawlik/gdmc_http_client_python) 5.0.

In time, some or all of these enhancements may be integrated into the main GDPC
package.

## Quick example

```python
buildArea = getBuildArea()

itf = Interface(buildArea.offset)

# Place a block
itf.placeBlock(ivec3(1,0,1), Block("grass_block"))

# Build a cube
placeBox(itf, Box(ivec3(4,0,0), ivec3(3,3,3)), Block("stone"))

# Build an oriented building in transformed local coordinates
transform = Transform(translation=ivec3(10,0,0), rotation=1, scale=ivec3(1,2,1))
with itf.pushTransform(transform):
    placeBox(itf, Box(size=ivec3(3,1,3)), Block("oak_log", axis="x"))

# Place a block with NBT data
nbt = 'Items: [{Slot: 13, id: "apple", Count: 1}]'
itf.placeBlock(ivec3(13,0,1), Block("chest", facing="south", nbt=nbt))

# Build a saved model, with any position and orientation, and with substitutions
models.testShape.build(
    itf,
    Transform(ivec3(16,0,0), rotation=3, scale=ivec3(1,1,-1)),
    substitutions={
        "minecraft:oak_planks": ["acacia_planks", "dark_oak_planks"]
    }
)
```

## Enhancements

### Vectors
This repository provides wrapped versions of many GDPC functions that work with
vectors (from `pyglm`), rather than separate x, y and z coordinates. Vectors
make many common math operations much easier. They are also the basis of many
of the more advanced enhancements listed below.

### Transformations
This repository's most important enhancement is probably its transformation
system. It allows you to "transform" your frame of reference for placing blocks,
so that you can always build using local coordinates instead of global ones. The
idea is based on the use of transformation matrices in typical 3D graphics
applications.

GDPC 5.0 already provides a basic transformation system: you can construct an
Interface class with x, y and z offsets. That is, GDPC 5.0 supports
*translations*. This repository, however, enhances this with 90-degree rotations
around the Y-axis and even integer scaling (and flipping, which is a special
case of scaling).

At the core of the transformation system lies the `Transform` class, which
essentially acts as a transformation matrix. It stores a translation (a 3D
vector), a rotation around the Y-axis (0, 1, 2 or 3) and an integer scale (a
3D vector). Transforms can be multiplied like matrices, and they can be applied
to vectors.

The `Block` class contains functionality that ensures that even *individual
blocks* which have an orientation (such as stairs) are rotated and flipped
correctly.

The transformation system allows you to write all your structure-building
code without needing to think about position or orientation (for example, doors
can always be at the bottom), since you can simply transform space to place the
structures anywhere you want. Transformations can even be stacked, allowing you
to use local coordinate systems within local coordinate systems.

### NBT data
Although GDPC 5.0 supports placing blocks with block state properties, it does
not fully support placing blocks with NBT data. This is mostly due to the fact
that the GDMC HTTP mod itself does not fully support it (though this may change
in the future). To set NBT data, you have to manually send a separate `/data`
command that modifies a block's NBT data. This is quite the hassle when using
local coordinates, since the `/data` command requires the global coordinates of
the block to change.

In this repository, the setting of NBT data is fully integrated via the `nbt`
field of the `Block` class. Whenever a block with NBT data is placed, an NBT-
setting command request is automatically sent, with the properly transformed
coordinates. If block placements are being buffered (a feature of GDPC to reduce
the amount of HTTP requests), these command requests are automatically deferred
until after the next buffer flush.

### Models
Often, it is easier to create a structure in Minecraft itself than it is to
define it in code. After all, Minecraft is a very complete and intuitive editor
for Minecraft. To support this workflow, this repository contains scripts that
allow you to scan in a "Model" of blocks from Minecraft and save it in a file
in the form of Python code that constructs an instance of the `Model` class.
Thanks to the transformation system, these models can then be placed anywhere
in the world. You could, for example, use them to build up large and complicated
structures from smaller model components.

The `Model` class also includes functionality to replace a block type with
another block type or a randomly sampled palette of block types when building.
This makes it possible to add random block palette textures to saved models, and
it allows you to create models that act only as a shape, with the specific
blocks to be chosen later. There is again a parallel with 3D graphics here:
shapes and textures are separated to increase modularity.

The `src/modelPull.py` script scans in a model from Minecraft and dumps it as
a string (with `xclip`, you can redirect the string to your clipboard:
`./modelPull.py | xclip -selection c`). You can store these model strings in a
Python file and then just import them as normal. See
[`src/models.py`](src/models.py) for an example. For more information on how
to use `modelPull.py`, use `./modelPull.py --help`.

The `src/modelBuild.py` script builds a model from a Python file (like
`src/models.py`) in the build area. It's a useful tool for model development.
Again, for more usage information, use `./modelBuild.py --help`.

### Asynchronous requests
Since blocks are placed using a HTTP interface, all placements are done via HTTP
requests. In GDPC 5.0, these requests are blocking: they freeze the program
until Minecraft has completed the placement request, and Minecraft can be quite
slow. GDPC 5.0 provides functionality to buffer block placements in order to
reduce this overhead, but while this improves performance significantly, block
placement remains very slow.

This repository integrates automatic asynchronous HTTP requests into the block
placing system. When enabled (with `Interface.multithreading = True`), all block
buffer flushes and NBT-setting requests are performed in a thread pool. The
effect on performance is varying: on some machines, it speeds up placement
extremely (x7 speedups have been observed), but on others it has no effect at
all.

Note that this system currently has an important limitation: when placing
multiple blocks at the same location, the order of placement may change. For
example, this can happen when first clearing an area with air and then placing a
structure in it. The amount of reversals seems to be surprisingly low, but it
can still sometimes cause issues.

It is currently **not** recommended to use this feature in production (i.e. in
a GDMC submission), but it may be useful to speed up your development.

### Miscellaneous additions
Besides the major features listed above, this repository contains many more
small additions, such as vector math utilities, additional geometry functions
and support for "no placement"-entries in block palettes.


## Installation

Currently, these enhancements are not available as a package. To use them,
simply copy the source files and the license file into your project and install
the requirements listed in [`requirements.txt`](requirements.txt). You can then
import the source files from your code as normal.

For information on how to use `modelPull.py` and `modelBuild.py`, run them
with a `--help` argument. Unfortunately, due to how Python's import system
works, these scripts need to lie in the root source directory.


## Detailed example code

[`src/example.py`](src/example.py) contains example code that demonstrates the
usage of most core enhancements. Note that it does assume some familiarity with
the base GDPC framework.
