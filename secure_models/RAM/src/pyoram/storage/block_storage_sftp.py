__all__ = ('BlockStorageSFTP',)

import logging

from pyoram.util.misc import chunkiter
from pyoram.storage.block_storage import \
     BlockStorageTypeFactory
from pyoram.storage.block_storage_file import \
    BlockStorageFile

log = logging.getLogger("pyoram")

class BlockStorageSFTP(BlockStorageFile):
    """
    A block storage device for accessing file data through
    an SSH portal using Secure File Transfer Protocol (SFTP).
    """

    def __init__(self,
                 storage_name,
                 sshclient=None,
                 **kwds):
        if sshclient is None:
            raise ValueError(
                "Can not open sftp block storage device "
                "without an ssh client.")
        super(BlockStorageSFTP, self).__init__(
            storage_name,
            _filesystem=sshclient.open_sftp(),
            **kwds)
        self._sshclient = sshclient
        self._f.set_pipelined()

    #
    # Define BlockStorageInterface Methods
    #

    def clone_device(self):
        f = BlockStorageSFTP(self.storage_name,
                             sshclient=self._sshclient,
                             threadpool_size=0,
                             ignore_lock=True)
        f._pool = self._pool
        f._close_pool = False
        return f

    #@classmethod
    #def compute_storage_size(...)

    @classmethod
    def setup(cls,
              storage_name,
              block_size,
              block_count,
              sshclient=None,
              threadpool_size=None,
              **kwds):
        if sshclient is None:
            raise ValueError(
                "Can not setup sftp block storage device "
                "without an ssh client.")

        with BlockStorageFile.setup(storage_name,
                                    block_size,
                                    block_count,
                                    _filesystem=sshclient.open_sftp(),
                                    threadpool_size=threadpool_size,
                                    **kwds) as f:
            pass
        f._filesystem.close()

        return BlockStorageSFTP(storage_name,
                                sshclient=sshclient,
                                threadpool_size=threadpool_size)

    #@property
    #def header_data(...)

    #@property
    #def block_count(...)

    #@property
    #def block_size(...)

    #@property
    #def storage_name(...)

    #def update_header_data(...)

    def close(self):
        super(BlockStorageSFTP, self).close()
        self._filesystem.close()

    def read_blocks(self, indices):
        self._check_async()
        args = []
        for i in indices:
            assert 0 <= i < self.block_count
            self._bytes_received += self.block_size
            args.append((self._header_offset + i * self.block_size,
                         self.block_size))
        return self._f.readv(args)

    def yield_blocks(self, indices, chunksize=100):
        for chunk in chunkiter(indices, n=chunksize):
            assert all(0 <= i <= self.block_count for i in chunk)
            self._bytes_received += self.block_size * len(chunk)
            args = [(self._header_offset + i * self.block_size,
                     self.block_size)
                    for i in chunk]
            for block in self._f.readv(args):
                yield block

    #def read_block(...)

    #def write_blocks(...)

    #def write_block(...)

    #@property
    #def bytes_sent(...)

    #@property
    #def bytes_received(...)

BlockStorageTypeFactory.register_device("sftp", BlockStorageSFTP)
