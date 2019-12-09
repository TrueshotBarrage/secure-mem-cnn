__all__ = ('BlockStorageRAM',)

import os
import struct
import logging
import errno
from multiprocessing.pool import ThreadPool

import pyoram
from pyoram.storage.block_storage import \
    (BlockStorageInterface,
     BlockStorageTypeFactory)
from pyoram.storage.block_storage_mmap import \
    (BlockStorageMMap,
     _BlockStorageMemoryImpl)

import tqdm
import six
from six.moves import xrange

log = logging.getLogger("pyoram")

class BlockStorageRAM(_BlockStorageMemoryImpl,
                      BlockStorageInterface):
    """
    A class implementing the block storage interface where all data is
    kept in RAM. This class uses the same storage format as
    BlockStorageFile. Thus, a block storage space can be created using
    this class and then, after saving the raw storage data to disk,
    reopened with any other class compatible with BlockStorageFile
    (and visa versa).
    """

    _index_struct_string = BlockStorageMMap._index_struct_string
    _index_offset = struct.calcsize(_index_struct_string)

    def __init__(self,
                 storage_data,
                 threadpool_size=None,
                 ignore_lock=False):

        self._bytes_sent = 0
        self._bytes_received = 0
        self._ignore_lock = ignore_lock
        self._f = None
        self._pool = None
        self._close_pool = True
        if type(storage_data) is not bytearray:
            raise TypeError(
                "BlockStorageRAM requires input argument of type "
                "'bytearray'. Invalid input type: %s"
                % (type(storage_data)))
        self._f = storage_data
        self._block_size, self._block_count, user_header_size, locked = \
            struct.unpack(
                BlockStorageRAM._index_struct_string,
                self._f[:BlockStorageRAM._index_offset])

        if locked and (not self._ignore_lock):
            raise IOError(
                "Can not open block storage device because it is "
                "locked by another process. To ignore this check, "
                "initialize this class with the keyword 'ignore_lock' "
                "set to True.")
        self._user_header_data = bytes()
        if user_header_size > 0:
            self._user_header_data = \
                bytes(self._f[BlockStorageRAM._index_offset:\
                              (BlockStorageRAM._index_offset+user_header_size)])
        assert len(self._user_header_data) == user_header_size
        self._header_offset = BlockStorageRAM._index_offset + \
                              len(self._user_header_data)

        if not self._ignore_lock:
            # turn on the locked flag
            self._f[:BlockStorageRAM._index_offset] = \
                struct.pack(BlockStorageRAM._index_struct_string,
                            self.block_size,
                            self.block_count,
                            len(self._user_header_data),
                            True)

        # Although we do not use the threadpool we still
        # create just in case we are the first
        if threadpool_size != 0:
            self._pool = ThreadPool(threadpool_size)

    #
    # Add some methods specific to BlockStorageRAM
    #

    @staticmethod
    def fromfile(file_,
                 threadpool_size=None,
                 ignore_lock=False):
        """
        Instantiate BlockStorageRAM device from a file saved in block
        storage format. The file_ argument can be a file object or a
        string that represents a filename. If called with a file
        object, it should be opened in binary mode, and the caller is
        responsible for closing the file.

        This method returns a BlockStorageRAM instance.
        """
        close_file = False
        if not hasattr(file_, 'read'):
            file_ = open(file_, 'rb')
            close_file = True
        try:
            header_data = file_.read(BlockStorageRAM._index_offset)
            block_size, block_count, user_header_size, locked = \
                struct.unpack(
                    BlockStorageRAM._index_struct_string,
                    header_data)
            if locked and (not ignore_lock):
                raise IOError(
                    "Can not open block storage device because it is "
                    "locked by another process. To ignore this check, "
                    "call this method with the keyword 'ignore_lock' "
                    "set to True.")
            header_offset = len(header_data) + \
                            user_header_size
            f = bytearray(header_offset + \
                          (block_size * block_count))
            f[:header_offset] = header_data + file_.read(user_header_size)
            f[header_offset:] = file_.read(block_size * block_count)
        finally:
            if close_file:
                file_.close()

        return BlockStorageRAM(f,
                               threadpool_size=threadpool_size,
                               ignore_lock=ignore_lock)

    def tofile(self, file_):
        """
        Dump all storage data to a file. The file_ argument can be a
        file object or a string that represents a filename. If called
        with a file object, it should be opened in binary mode, and
        the caller is responsible for closing the file.

        The method should only be called after the storage device has
        been closed to ensure that the locked flag has been set to
        False.
        """
        close_file = False
        if not hasattr(file_, 'write'):
            file_ = open(file_, 'wb')
            close_file = True
        file_.write(self._f)
        if close_file:
            file_.close()

    @property
    def data(self):
        """Access the raw bytearray"""
        return self._f

    #
    # Define BlockStorageInterface Methods
    #

    def clone_device(self):
        f = BlockStorageRAM(self._f,
                            threadpool_size=0,
                            ignore_lock=True)
        f._pool = self._pool
        f._close_pool = False
        return f

    @classmethod
    def compute_storage_size(cls, *args, **kwds):
        return BlockStorageMMap.compute_storage_size(*args, **kwds)

    @classmethod
    def setup(cls,
              storage_name,
              block_size,
              block_count,
              initialize=None,
              header_data=None,
              ignore_existing=False,
              threadpool_size=None):

        # We ignore the 'storage_name' argument
        # We ignore the 'ignore_existing' flag
        if (block_size <= 0) or (block_size != int(block_size)):
            raise ValueError(
                "Block size (bytes) must be a positive integer: %s"
                % (block_size))
        if (block_count <= 0) or (block_count != int(block_count)):
            raise ValueError(
                "Block count must be a positive integer: %s"
                % (block_count))
        if (header_data is not None) and \
           (type(header_data) is not bytes):
            raise TypeError(
                "'header_data' must be of type bytes. "
                "Invalid type: %s" % (type(header_data)))

        if initialize is None:
            zeros = bytes(bytearray(block_size))
            initialize = lambda i: zeros

        # create_index
        index_data = None
        if header_data is None:
            index_data = struct.pack(BlockStorageRAM._index_struct_string,
                                     block_size,
                                     block_count,
                                     0,
                                     False)
            header_data = bytes()
        else:
            index_data = struct.pack(BlockStorageRAM._index_struct_string,
                                     block_size,
                                     block_count,
                                     len(header_data),
                                     False)
        header_offset = len(index_data) + len(header_data)
        f = bytearray(header_offset + \
                      (block_size * block_count))
        f[:header_offset] = index_data + header_data
        progress_bar = tqdm.tqdm(total=block_count*block_size,
                                 desc="Initializing File Block Storage Space",
                                 unit="B",
                                 unit_scale=True,
                                 disable=not pyoram.config.SHOW_PROGRESS_BAR)
        for i in xrange(block_count):
            block = initialize(i)
            assert len(block) == block_size, \
                ("%s != %s" % (len(block), block_size))
            pos_start = header_offset + i * block_size
            pos_start = header_offset + i * block_size
            pos_stop = pos_start + block_size
            f[pos_start:pos_stop] = block[:]
            progress_bar.update(n=block_size)
        progress_bar.close()

        return BlockStorageRAM(f, threadpool_size=threadpool_size)

    @property
    def header_data(self):
        return self._user_header_data

    @property
    def block_count(self):
        return self._block_count

    @property
    def block_size(self):
        return self._block_size

    @property
    def storage_name(self):
        return None

    def update_header_data(self, new_header_data):
        if len(new_header_data) != len(self.header_data):
            raise ValueError(
                "The size of header data can not change.\n"
                "Original bytes: %s\n"
                "New bytes: %s" % (len(self.header_data),
                                   len(new_header_data)))
        self._user_header_data = bytes(new_header_data)
        self._f[BlockStorageRAM._index_offset:\
                (BlockStorageRAM._index_offset+len(new_header_data))] = \
            self._user_header_data

    def close(self):
        if self._close_pool and (self._pool is not None):
            self._pool.close()
            self._pool.join()
            self._pool = None
        if not self._ignore_lock:
            # turn off the locked flag
            self._f[:BlockStorageRAM._index_offset] = \
                struct.pack(BlockStorageRAM._index_struct_string,
                            self.block_size,
                            self.block_count,
                            len(self._user_header_data),
                            False)
            self._ignore_lock = True

    #
    # We must cast from bytearray to bytes
    # when reading from a bytearray so that this
    # class works with the encryption layer.
    #

    def read_blocks(self, indices):
        return [bytes(block) for block
                in super(BlockStorageRAM, self).read_blocks(indices)]

    def yield_blocks(self, indices):
        for block in super(BlockStorageRAM, self).yield_blocks(indices):
            yield bytes(block)

    def read_block(self, i):
        return bytes(super(BlockStorageRAM, self).read_block(i))

    #def write_blocks(...)

    #def write_block(...)

    @property
    def bytes_sent(self):
        return self._bytes_sent

    @property
    def bytes_received(self):
        return self._bytes_received

BlockStorageTypeFactory.register_device("ram", BlockStorageRAM)
