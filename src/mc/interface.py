"""
Wrappers around gdpc v5.0 that work with vectors
"""

from typing import Union, Optional, List
from copy import copy
import random
import time
from concurrent import futures
from glm import ivec3, bvec2

from util.util import isign, stdoutToStderr, eprint

with stdoutToStderr(): # GDPC outputting to stdout on import messes with some scripts
    from gdpc import interface, direct_interface, worldLoader

from .vectorUtil import Transform, Rect, areaBetween
from .block import Block


def getBuildArea():
    with stdoutToStderr():
        beginX, beginY, beginZ, endX, endY, endZ = interface.requestBuildArea()
    return areaBetween(ivec3(beginX, beginY, beginZ), ivec3(endX, endY, endZ))


def getWorldSlice(rect: Rect):
    assert isinstance(rect, Rect) # To protect from calling this with an Area
    attempts = 0
    while True:
        try:
            attempts += 1
            return worldLoader.WorldSlice(rect.begin[0], rect.begin[1], rect.end[0], rect.end[1])
        except Exception as e: # pylint: disable=broad-except
            if attempts < 10:
                print("Could not get the world slice. Try reducing your render distance. I'll retry in a bit.")
                time.sleep(2)
            else:
                print("OK, that's enough retries. You deal with the exception.")
                raise


def run_command(command: str):
    """ Executes one or multiple minecraft commands (separated by newlines) """
    # eprint("running cmd " + command)
    direct_interface.runCommand(command)


def block_nbt_command(position: ivec3, nbt: str):
    """ Returns the command required to merge the nbt data of the block at the global position
        [position] with [nbt] """
    return f"data merge block {position.x} {position.y} {position.z} {{{nbt}}}"


class Interface:
    """ Wrapper around gdpc v5.0's Interface class """
    def __init__(
        self,
        transform             = Transform(),
        buffering             = True,
        bufferLimit           = 1024,
        multithreading        = False,
        multithreadingWorkers = 8,
        caching               = False,
        cacheLimit            = 8192,
    ):
        self.transform = transform
        self.gdpcInterface = interface.Interface(
            buffering   = buffering,
            bufferlimit = bufferLimit + 1, # +1 so we can intercept the buffer flushes
            caching     = caching,
            cachelimit  = cacheLimit
        )
        self.__multithreadingWorkers = multithreadingWorkers
        self.__buffer_flush_executor = None
        self.multithreading = multithreading # Creates the buffer flush executor if True
        self.__buffer_flush_futures: List[futures.Future] = []
        self.__command_buffer:       List[str]            = []


    @property
    def buffering(self) -> bool:
        return self.gdpcInterface.isBuffering()

    @buffering.setter
    def buffering(self, value: bool):
        if self.buffering and not value:
            self.sendBufferedBlocks()
        self.gdpcInterface.setBuffering(value, notify=False)

    @property
    def bufferLimit(self) -> int:
        return self.gdpcInterface.getBufferLimit() - 1

    @bufferLimit.setter
    def bufferLimit(self, value: int):
        self.gdpcInterface.setBufferLimit(value + 1)

    @property
    def caching(self):
        return self.gdpcInterface.isCaching()

    @caching.setter
    def caching(self, value: bool):
        self.gdpcInterface.setCaching(value)

    @property
    def cacheLimit(self):
        return self.gdpcInterface.getCacheLimit()

    @cacheLimit.setter
    def cacheLimit(self, value: int):
        self.gdpcInterface.setCacheLimit()

    @property
    def multithreading(self):
        return self.__multithreading

    @multithreading.setter
    def multithreading(self, value: bool):
        self.__multithreading = value
        if value and self.__buffer_flush_executor is None:
            self.__buffer_flush_executor = futures.ThreadPoolExecutor(self.__multithreadingWorkers)

    # Modifying the amount of workers is not supported
    @property
    def multithreadingWorkers(self):
        return self.__multithreadingWorkers


    def run_command(self, command: str):
        """ Executes one or multiple minecraft commands (separated by newlines).
            If buffering is enabled, the command is deferred until after the next buffer flush. """
        if self.buffering:
            self.__command_buffer.append(command)
        else:
            run_command(command)


    def localToGlobal(self, t: Transform):
        return self.transform.compose(t)

    def globalToLocal(self, t: Transform):
        return self.transform.invCompose(t)


    def placeStringGlobal(
        self,
        position:     ivec3,
        scale:        ivec3, # Note that any flipping must be pre-applied to [block_state]!
        block_string: Union[str, List[str]],
        block_state:  str = "",
        nbt:          Optional[str] = None,
        replace:      Optional[Union[str, List[str]]] = None,
    ):
        with stdoutToStderr():
            for x in range(position.x, position.x + scale.x, isign(scale.x)):
                for y in range(position.y, position.y + scale.y, isign(scale.y)):
                    for z in range(position.z, position.z + scale.z, isign(scale.z)):
                        # Select block from palette
                        if isinstance(block_string, str):
                            chosen_block_string = block_string
                        else:
                            chosen_block_string = random.choice(block_string)
                        # Place the block
                        if chosen_block_string != "": # Support for "nothing" in palettes
                            self.gdpcInterface.placeBlock(
                                x, y, z,
                                chosen_block_string + block_state,
                                replace
                            )

        if nbt is not None:
            self.run_command(block_nbt_command(position, nbt))

        # Redirect all buffer flushes to self.sendBufferedBlocks, so we can implement automatic
        # multithreaded buffer flushing.
        if len(self.gdpcInterface.buffer) == self.bufferLimit:
            self.sendBufferedBlocks()


    def placeBlockGlobal(
        self,
        position: ivec3,
        rotation: int,
        scale:    ivec3,
        block:    Block,
        replace:  Optional[Union[str, List[str]]] = None,
    ):
        self.placeStringGlobal(
            position,
            scale,
            block.name,
            block.blockStateString(rotation, bvec2(scale.x < 0, scale.z < 0)),
            block.nbt,
            replace
        )


    def placeBlock(
        self,
        t:       Union[Transform, ivec3],
        block:   Block,
        replace: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(t, ivec3): t = Transform(t)
        gt = self.localToGlobal(t)
        self.placeBlockGlobal(gt.translation, gt.rotation, gt.scale, block, replace)


    def sendBufferedBlocks(self, retries = 5):
        """ If multithreaded buffer flushes are enabled, they can be awaited with
            awaitBufferFlushes(). """

        if self.__multithreading:

            # Clean up finished buffer flush futures
            self.__buffer_flush_futures = [
                future for future in self.__buffer_flush_futures if not future.done()
            ]

            # Shallow copies are good enough here
            gdpc_interface_copy = copy(self.gdpcInterface)
            command_buffer_copy = copy(self.__command_buffer)
            def task():
                with stdoutToStderr():
                    gdpc_interface_copy.sendBlocks(retries=retries)
                for command in command_buffer_copy:
                    run_command(command)

            # Submit the task
            future = self.__buffer_flush_executor.submit(task)
            self.__buffer_flush_futures.append(future)

            # Empty the buffers (the thread has copies of the references)
            self.gdpcInterface.buffer = []
            self.__command_buffer     = []

        else: # No multithreading

            with stdoutToStderr():
                self.gdpcInterface.sendBlocks(retries=retries)
            for command in self.__command_buffer:
                run_command(command)
            self.__command_buffer = []


    def awaitBufferFlushes(self, timeout: Optional[float] = None):
        """ Await all pending buffer flushes """
        self.__buffer_flush_futures = futures.wait(self.__buffer_flush_futures, timeout).not_done


    # TODO: if need be, we can wrap gdpcInterface.getBlock() as well.
