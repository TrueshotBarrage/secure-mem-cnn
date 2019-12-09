__all__ = ('BlockStorageS3',)

import struct
import logging
from multiprocessing.pool import ThreadPool

import pyoram
from pyoram.storage.block_storage import \
    (BlockStorageInterface,
     BlockStorageTypeFactory)
from pyoram.storage.boto3_s3_wrapper import Boto3S3Wrapper

import tqdm
import six
from six.moves import xrange, map

log = logging.getLogger("pyoram")

class BlockStorageS3(BlockStorageInterface):
    """
    A block storage device for Amazon Simple
    Storage Service (S3).
    """

    _index_name = "PyORAMBlockStorageS3_index.bin"
    _index_struct_string = "!LLL?"
    _index_offset = struct.calcsize(_index_struct_string)

    def __init__(self,
                 storage_name,
                 bucket_name=None,
                 aws_access_key_id=None,
                 aws_secret_access_key=None,
                 region_name=None,
                 ignore_lock=False,
                 threadpool_size=None,
                 s3_wrapper=Boto3S3Wrapper):

        self._bytes_sent = 0
        self._bytes_received = 0
        self._storage_name = storage_name
        self._bucket_name = bucket_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._region_name = region_name
        self._pool = None
        self._close_pool = True
        self._s3 = None
        self._ignore_lock = ignore_lock
        self._async_write = None
        self._async_write_callback = None

        if bucket_name is None:
            raise ValueError("'bucket_name' keyword is required")

        if threadpool_size != 0:
            self._pool = ThreadPool(threadpool_size)

        self._s3 = s3_wrapper(bucket_name,
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key,
                              region_name=region_name)
        self._basename = self.storage_name+"/b%d"

        index_data = self._s3.download(
            self._storage_name+"/"+BlockStorageS3._index_name)
        self._block_size, self._block_count, user_header_size, locked = \
            struct.unpack(
                BlockStorageS3._index_struct_string,
                index_data[:BlockStorageS3._index_offset])
        if locked and (not self._ignore_lock):
            raise IOError(
                "Can not open block storage device because it is "
                "locked by another process. To ignore this check, "
                "initialize this class with the keyword 'ignore_lock' "
                "set to True.")
        self._user_header_data = bytes()
        if user_header_size > 0:
            self._user_header_data = \
                index_data[BlockStorageS3._index_offset:
                           (BlockStorageS3._index_offset+user_header_size)]

        if not self._ignore_lock:
            # turn on the locked flag
            self._s3.upload((self._storage_name+"/"+BlockStorageS3._index_name,
                             struct.pack(BlockStorageS3._index_struct_string,
                                         self.block_size,
                                         self.block_count,
                                         len(self.header_data),
                                         True) + \
                             self.header_data))

    def _check_async(self):
        if self._async_write is not None:
            for i in self._async_write:
                if self._async_write_callback is not None:
                    self._async_write_callback(i)
            self._async_write = None
            self._async_write_callback = None

    def _schedule_async_write(self, arglist, callback=None):
        assert self._async_write is None
        if self._pool is not None:
            self._async_write = \
                self._pool.imap_unordered(self._s3.upload, arglist)
            self._async_write_callback = callback
        else:
            # Note: we are using six.map which always
            #       behaves like imap
            for i in map(self._s3.upload, arglist):
                if callback is not None:
                    callback(i)

    def _download(self, i):
        return self._s3.download(self._basename % i)

    #
    # Define BlockStorageInterface Methods
    #

    def clone_device(self):
        f =  BlockStorageS3(self.storage_name,
                            bucket_name=self._bucket_name,
                            aws_access_key_id=self._aws_access_key_id,
                            aws_secret_access_key=self._aws_secret_access_key,
                            region_name=self._region_name,
                            threadpool_size=0,
                            s3_wrapper=type(self._s3),
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
            return BlockStorageS3._index_offset + \
                    len(header_data) + \
                    block_size * block_count

    @classmethod
    def setup(cls,
              storage_name,
              block_size,
              block_count,
              bucket_name=None,
              aws_access_key_id=None,
              aws_secret_access_key=None,
              region_name=None,
              header_data=None,
              initialize=None,
              threadpool_size=None,
              ignore_existing=False,
              s3_wrapper=Boto3S3Wrapper):

        if bucket_name is None:
            raise ValueError("'bucket_name' is required")
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

        pool = None
        if threadpool_size != 0:
            pool = ThreadPool(threadpool_size)

        s3 = s3_wrapper(bucket_name,
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region_name=region_name)
        exists = s3.exists(storage_name)
        if (not ignore_existing) and exists:
            raise IOError(
                "Storage location already exists in bucket %s: %s"
                % (bucket_name, storage_name))
        if exists:
            log.info("Deleting objects in existing S3 entry: %s/%s"
                     % (bucket_name, storage_name))
            print("Clearing Existing S3 Objects With Prefix %s/%s/"
                  % (bucket_name, storage_name))
            s3.clear(storage_name, threadpool=pool)

        if header_data is None:
            s3.upload((storage_name+"/"+BlockStorageS3._index_name,
                       struct.pack(BlockStorageS3._index_struct_string,
                                  block_size,
                                  block_count,
                                  0,
                                  False)))
        else:
            s3.upload((storage_name+"/"+BlockStorageS3._index_name,
                       struct.pack(BlockStorageS3._index_struct_string,
                                   block_size,
                                   block_count,
                                   len(header_data),
                                   False) + \
                       header_data))

        if initialize is None:
            zeros = bytes(bytearray(block_size))
            initialize = lambda i: zeros
        basename = storage_name+"/b%d"
        # NOTE: We will not be informed when a thread
        #       encounters an exception (e.g., when
        #       calling initialize(i). We must ensure
        #       that all iterations were processed
        #       by counting the results.
        def init_blocks():
            for i in xrange(block_count):
                yield (basename % i, initialize(i))
        def _do_upload(arg):
            try:
                s3.upload(arg)
            except Exception as e:                     # pragma: no cover
                log.error(                             # pragma: no cover
                    "An exception occured during S3 "  # pragma: no cover
                    "setup when calling the block "    # pragma: no cover
                    "initialization function: %s"      # pragma: no cover
                    % (str(e)))                        # pragma: no cover
                raise                                  # pragma: no cover
        total = None
        progress_bar = tqdm.tqdm(total=block_count*block_size,
                                 desc="Initializing S3 Block Storage Space",
                                 unit="B",
                                 unit_scale=True,
                                 disable=not pyoram.config.SHOW_PROGRESS_BAR)
        if pool is not None:
            try:
                for i,_ in enumerate(
                        pool.imap_unordered(_do_upload, init_blocks())):
                    total = i
                    progress_bar.update(n=block_size)
            except Exception as e:                     # pragma: no cover
                s3.clear(storage_name)                 # pragma: no cover
                raise                                  # pragma: no cover
            finally:
                progress_bar.close()
                pool.close()
                pool.join()
        else:
            try:
                for i,_ in enumerate(
                        map(s3.upload, init_blocks())):
                    total = i
                    progress_bar.update(n=block_size)
            except Exception as e:                     # pragma: no cover
                s3.clear(storage_name)                 # pragma: no cover
                raise                                  # pragma: no cover
            finally:
                progress_bar.close()

        if total != block_count - 1:
            s3.clear(storage_name)                     # pragma: no cover
            if pool is not None:                       # pragma: no cover
                pool.close()                           # pragma: no cover
                pool.join()                            # pragma: no cover
            raise ValueError(                          # pragma: no cover
                "Something went wrong during S3 block" # pragma: no cover
                " initialization. Check the logger "   # pragma: no cover
                "output for more information.")        # pragma: no cover

        return BlockStorageS3(storage_name,
                              bucket_name=bucket_name,
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key,
                              region_name=region_name,
                              threadpool_size=threadpool_size,
                              s3_wrapper=s3_wrapper)

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
        self._user_header_data = new_header_data

        index_data = bytearray(self._s3.download(
            self._storage_name+"/"+BlockStorageS3._index_name))
        lenbefore = len(index_data)
        index_data[BlockStorageS3._index_offset:] = new_header_data
        assert lenbefore == len(index_data)
        self._s3.upload((self._storage_name+"/"+BlockStorageS3._index_name,
                         bytes(index_data)))

    def close(self):
        self._check_async()
        if self._s3 is not None:
            if not self._ignore_lock:
                # turn off the locked flag
                self._s3.upload(
                    (self._storage_name+"/"+BlockStorageS3._index_name,
                     struct.pack(BlockStorageS3._index_struct_string,
                                 self.block_size,
                                 self.block_count,
                                 len(self.header_data),
                                 False) + \
                     self.header_data))
        if self._close_pool and (self._pool is not None):
            self._pool.close()
            self._pool.join()
            self._pool = None

    def read_blocks(self, indices):
        self._check_async()
        # be sure not to exhaust this if it is an iterator
        # or generator
        indices = list(indices)
        assert all(0 <= i <= self.block_count for i in indices)
        self._bytes_received += self.block_size * len(indices)
        if self._pool is not None:
            return self._pool.map(self._download, indices)
        else:
            return list(map(self._download, indices))

    def yield_blocks(self, indices):
        self._check_async()
        # be sure not to exhaust this if it is an iterator
        # or generator
        indices = list(indices)
        assert all(0 <= i <= self.block_count for i in indices)
        self._bytes_received += self.block_size * len(indices)
        if self._pool is not None:
            return self._pool.imap(self._download, indices)
        else:
            return map(self._download, indices)

    def read_block(self, i):
        self._check_async()
        assert 0 <= i < self.block_count
        self._bytes_received += self.block_size
        return self._download(i)

    def write_blocks(self, indices, blocks, callback=None):
        self._check_async()
        # be sure not to exhaust this if it is an iterator
        # or generator
        indices = list(indices)
        assert all(0 <= i <= self.block_count for i in indices)
        self._bytes_sent += self.block_size * len(indices)
        indices = (self._basename % i for i in indices)
        self._schedule_async_write(zip(indices, blocks),
                                   callback=callback)

    def write_block(self, i, block):
        self._check_async()
        assert 0 <= i < self.block_count
        self._bytes_sent += self.block_size
        self._schedule_async_write((((self._basename % i), block),))

    @property
    def bytes_sent(self):
        return self._bytes_sent

    @property
    def bytes_received(self):
        return self._bytes_received

BlockStorageTypeFactory.register_device("s3", BlockStorageS3)
