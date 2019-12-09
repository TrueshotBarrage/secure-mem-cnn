__all__ = ('BlockStorageFile',)

import os
import struct
import logging
import errno
from multiprocessing.pool import ThreadPool

import pyoram
from pyoram.storage.block_storage import \
    (BlockStorageInterface,
     BlockStorageTypeFactory)

import tqdm
import six
from six.moves import xrange

log = logging.getLogger("pyoram")

class default_filesystem(object):
    open = open
    remove = os.remove
    stat = os.stat

class BlockStorageFile(BlockStorageInterface):
    """
    A class implementing the block storage interface
    using a local file.
    """

    _index_struct_string = "!LLL?"
    _index_offset = struct.calcsize(_index_struct_string)

    def __init__(self,
                 storage_name,
                 threadpool_size=None,
                 ignore_lock=False,
                 _filesystem=default_filesystem):

        self._bytes_sent = 0
        self._bytes_received = 0
        self._filesystem = _filesystem
        self._ignore_lock = ignore_lock
        self._f = None
        self._pool = None
        self._close_pool = True
        self._async_write = None
        self._storage_name = storage_name
        self._f = self._filesystem.open(self.storage_name, "r+b")
        self._f.seek(0)
        self._block_size, self._block_count, user_header_size, locked = \
            struct.unpack(
                BlockStorageFile._index_struct_string,
                self._f.read(BlockStorageFile._index_offset))

        if locked and (not self._ignore_lock):
            self._f.close()
            self._f = None
            raise IOError(
                "Can not open block storage device because it is "
                "locked by another process. To ignore this check, "
                "initialize this class with the keyword 'ignore_lock' "
                "set to True.")
        self._user_header_data = bytes()
        if user_header_size > 0:
            self._user_header_data = \
                self._f.read(user_header_size)
        self._header_offset = BlockStorageFile._index_offset + \
                              len(self._user_header_data)

        # TODO: Figure out why this is required for Python3
        #       in order to prevent issues with the
        #       TopCachedEncryptedHeapStorage class. The
        #       problem has something to do with bufferedio,
        #       but it makes no sense why this fixes it (all
        #       we've done is read above these lines). As
        #       part of this, investigate whethor or not we
        #       need the call to flush after write_block(s),
        #       or if its simply connected to some Python3
        #       bug in bufferedio.
        self._f.flush()

        if not self._ignore_lock:
            # turn on the locked flag
            self._f.seek(0)
            self._f.write(
                struct.pack(BlockStorageFile._index_struct_string,
                            self.block_size,
                            self.block_count,
                            len(self._user_header_data),
                            True))
            self._f.flush()

        if threadpool_size != 0:
            self._pool = ThreadPool(threadpool_size)

    def _check_async(self):
        if self._async_write is not None:
            self._async_write.get()
            self._async_write = None
        # TODO: Figure out why tests fail on Python3 without this
        if six.PY3:
            if self._f is None:
                return
            self._f.flush()

    def _schedule_async_write(self, args, callback=None):
        assert self._async_write is None
        if self._pool is not None:
            self._async_write = \
                self._pool.apply_async(self._writev, (args, callback))
        else:
            self._writev(args, callback)

    # This method is usually executed in another thread, so
    # do not attempt to handle exceptions because it will
    # not work.
    def _writev(self, chunks, callback):
        for i, block in chunks:
            self._f.seek(self._header_offset + i * self.block_size)
            self._f.write(block)
            if callback is not None:
                callback(i)

    def _prep_for_close(self):
        self._check_async()
        if self._close_pool and (self._pool is not None):
            self._pool.close()
            self._pool.join()
            self._pool = None
        if self._f is not None:
            if not self._ignore_lock:
                # turn off the locked flag
                self._f.seek(0)
                self._f.write(
                    struct.pack(BlockStorageFile._index_struct_string,
                                self.block_size,
                                self.block_count,
                                len(self._user_header_data),
                                False))
                self._f.flush()

    #
    # Define BlockStorageInterface Methods
    #

    def clone_device(self):
        f = BlockStorageFile(self.storage_name,
                             threadpool_size=0,
                             ignore_lock=True)
        f._pool = self._pool
        f._close_pool = False
        return f

    @classmethod
    def compute_storage_size(cls,
                             block_size,
                             block_count,
                             header_data=None,
                             ignore_header=False):
        assert (block_size > 0) and (block_size == int(block_size))
        assert (block_count > 0) and (block_count == int(block_count))
        if header_data is None:
            header_data = bytes()
        if ignore_header:
            return block_size * block_count
        else:
            return BlockStorageFile._index_offset + \
                   len(header_data) + \
                   block_size * block_count

    @classmethod
    def setup(cls,
              storage_name,
              block_size,
              block_count,
              initialize=None,
              header_data=None,
              ignore_existing=False,
              threadpool_size=None,
              _filesystem=default_filesystem):

        if (not ignore_existing):
            _exists = True
            try:
                _filesystem.stat(storage_name)
            except OSError as e:
                if e.errno == errno.ENOENT:
                    _exists = False
            if _exists:
                raise IOError(
                    "Storage location already exists: %s"
                    % (storage_name))
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
        try:
            with _filesystem.open(storage_name, "wb") as f:
                # create_index
                if header_data is None:
                    f.write(struct.pack(BlockStorageFile._index_struct_string,
                                        block_size,
                                        block_count,
                                        0,
                                        False))
                else:
                    f.write(struct.pack(BlockStorageFile._index_struct_string,
                                        block_size,
                                        block_count,
                                        len(header_data),
                                        False))
                    f.write(header_data)
                with tqdm.tqdm(total=block_count*block_size,
                               desc="Initializing File Block Storage Space",
                               unit="B",
                               unit_scale=True,
                               disable=not pyoram.config.SHOW_PROGRESS_BAR) as progress_bar:
                    for i in xrange(block_count):
                        block = initialize(i)
                        assert len(block) == block_size, \
                            ("%s != %s" % (len(block), block_size))
                        f.write(block)
                        progress_bar.update(n=block_size)
        except:                                        # pragma: no cover
            _filesystem.remove(storage_name)           # pragma: no cover
            raise                                      # pragma: no cover

        return BlockStorageFile(storage_name,
                                threadpool_size=threadpool_size,
                                _filesystem=_filesystem)

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
        return self._storage_name

    def update_header_data(self, new_header_data):
        self._check_async()
        if len(new_header_data) != len(self.header_data):
            raise ValueError(
                "The size of header data can not change.\n"
                "Original bytes: %s\n"
                "New bytes: %s" % (len(self.header_data),
                                   len(new_header_data)))
        self._user_header_data = bytes(new_header_data)
        self._f.seek(BlockStorageFile._index_offset)
        self._f.write(self._user_header_data)

    def close(self):
        self._prep_for_close()
        if self._f is not None:
            try:
                self._f.close()
            except OSError:                            # pragma: no cover
                pass                                   # pragma: no cover
            self._f = None

    def read_blocks(self, indices):
        self._check_async()
        blocks = []
        for i in indices:
            assert 0 <= i < self.block_count
            self._bytes_received += self.block_size
            self._f.seek(self._header_offset + i * self.block_size)
            blocks.append(self._f.read(self.block_size))
        return blocks

    def yield_blocks(self, indices):
        self._check_async()
        for i in indices:
            assert 0 <= i < self.block_count
            self._bytes_received += self.block_size
            self._f.seek(self._header_offset + i * self.block_size)
            yield self._f.read(self.block_size)

    def read_block(self, i):
        self._check_async()
        assert 0 <= i < self.block_count
        self._bytes_received += self.block_size
        self._f.seek(self._header_offset + i * self.block_size)
        return self._f.read(self.block_size)

    def write_blocks(self, indices, blocks, callback=None):
        self._check_async()
        chunks = []
        for i, block in zip(indices, blocks):
            assert 0 <= i < self.block_count
            assert len(block) == self.block_size, \
                ("%s != %s" % (len(block), self.block_size))
            self._bytes_sent += self.block_size
            chunks.append((i, block))
        self._schedule_async_write(chunks, callback=callback)

    def write_block(self, i, block):
        self._check_async()
        assert 0 <= i < self.block_count
        assert len(block) == self.block_size
        self._bytes_sent += self.block_size
        self._schedule_async_write(((i, block),))

    @property
    def bytes_sent(self):
        return self._bytes_sent

    @property
    def bytes_received(self):
        return self._bytes_received

BlockStorageTypeFactory.register_device("file", BlockStorageFile)
