__all__ = ('BlockStorageMMap',)

import logging
import mmap

from pyoram.storage.block_storage import \
    BlockStorageTypeFactory
from pyoram.storage.block_storage_file import \
    BlockStorageFile

log = logging.getLogger("pyoram")

class _BlockStorageMemoryImpl(object):
    """
    This class implementents the BlockStorageInterface read/write
    methods for classes with a private attribute _f that can be
    accessed using __getslice__/__setslice__ notation.
    """

    def read_blocks(self, indices):
        blocks = []
        for i in indices:
            assert 0 <= i < self.block_count
            self._bytes_received += self.block_size
            pos_start = self._header_offset + i * self.block_size
            pos_stop = pos_start + self.block_size
            blocks.append(self._f[pos_start:pos_stop])
        return blocks

    def yield_blocks(self, indices):
        for i in indices:
            assert 0 <= i < self.block_count
            self._bytes_received += self.block_size
            pos_start = self._header_offset + i * self.block_size
            pos_stop = pos_start + self.block_size
            yield self._f[pos_start:pos_stop]

    def read_block(self, i):
        assert 0 <= i < self.block_count
        self._bytes_received += self.block_size
        pos_start = self._header_offset + i * self.block_size
        pos_stop = pos_start + self.block_size
        return self._f[pos_start:pos_stop]

    def write_blocks(self, indices, blocks, callback=None):
        for i, block in zip(indices, blocks):
            assert 0 <= i < self.block_count
            self._bytes_sent += self.block_size
            pos_start = self._header_offset + i * self.block_size
            pos_stop = pos_start + self.block_size
            self._f[pos_start:pos_stop] = block
            if callback is not None:
                callback(i)

    def write_block(self, i, block):
        assert 0 <= i < self.block_count
        self._bytes_sent += self.block_size
        pos_start = self._header_offset + i * self.block_size
        pos_stop = pos_start + self.block_size
        self._f[pos_start:pos_stop] = block

class BlockStorageMMap(_BlockStorageMemoryImpl,
                       BlockStorageFile):
    """
    A class implementing the block storage interface by creating a
    memory map over a local file. This class uses the same storage
    format as BlockStorageFile. Thus, a block storage space can be
    created using this class and then, after saving the raw storage
    data to disk, reopened with any other class compatible with
    BlockStorageFile (and visa versa).
    """

    def __init__(self, *args, **kwds):
        mm = kwds.pop('mm', None)
        self._mmap_owned = True
        super(BlockStorageMMap, self).__init__(*args, **kwds)
        if mm is None:
            self._f.flush()
            mm = mmap.mmap(self._f.fileno(), 0)
        else:
            self._mmap_owned = False
        self._f.close()
        self._f = mm

    #
    # Define BlockStorageInterface Methods
    # (override what is defined on BlockStorageFile)
    #

    #@classmethod
    #def compute_storage_size(...)

    def clone_device(self):
        f =  BlockStorageMMap(self.storage_name,
                              threadpool_size=0,
                              mm=self._f,
                              ignore_lock=True)
        f._pool = self._pool
        f._close_pool = False
        return f

    @classmethod
    def setup(cls,
              storage_name,
              block_size,
              block_count,
              **kwds):
        f = BlockStorageFile.setup(storage_name,
                                   block_size,
                                   block_count,
                                   **kwds)
        f.close()
        return BlockStorageMMap(storage_name)

    #def update_header_data(...)

    def close(self):
        self._prep_for_close()
        if self._f is not None:
            if self._mmap_owned:
                try:
                    self._f.close()
                except OSError:                        # pragma: no cover
                    pass                               # pragma: no cover
            self._f = None

    #def read_blocks(...)

    #def yield_blocks(...)

    #def read_block(...)

    #def write_blocks(...)

    #def write_block(...)

    #@property
    #def bytes_sent(...)

    #@property
    #def bytes_received(...)

BlockStorageTypeFactory.register_device("mmap", BlockStorageMMap)
