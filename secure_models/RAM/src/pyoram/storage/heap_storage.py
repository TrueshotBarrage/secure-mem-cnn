__all__ = ('HeapStorage',)

import struct

from pyoram.util.virtual_heap import SizedVirtualHeap
from pyoram.storage.block_storage import (BlockStorageInterface,
                                          BlockStorageTypeFactory)

class HeapStorageInterface(object):

    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()

    #
    # Abstract Interface
    #

    def clone_device(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover

    @classmethod
    def compute_storage_size(cls, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @classmethod
    def setup(cls, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover

    @property
    def header_data(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def bucket_count(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def bucket_size(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def blocks_per_bucket(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def storage_name(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def virtual_heap(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def bucket_storage(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover

    def update_header_data(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    def close(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    def read_path(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    def write_path(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover

    @property
    def bytes_sent(self):
        raise NotImplementedError                      # pragma: no cover
    @property
    def bytes_received(self):
        raise NotImplementedError                      # pragma: no cover

class HeapStorage(HeapStorageInterface):

    _header_struct_string = "!LLL"
    _header_offset = struct.calcsize(_header_struct_string)

    def _new_storage(self, storage, **kwds):
        storage_type = kwds.pop('storage_type', 'file')


    def __init__(self, storage, **kwds):
        if isinstance(storage, BlockStorageInterface):
            self._storage = storage
            if len(kwds):
                raise ValueError(
                    "Keywords not used when initializing "
                    "with a storage device: %s"
                    % (str(kwds)))
        else:
            storage_type = kwds.pop('storage_type', 'file')
            self._storage = BlockStorageTypeFactory(storage_type)\
                            (storage, **kwds)

        heap_base, heap_height, blocks_per_bucket = \
            struct.unpack(
                self._header_struct_string,
                self._storage.header_data[:self._header_offset])
        self._vheap = SizedVirtualHeap(
            heap_base,
            heap_height,
            blocks_per_bucket=blocks_per_bucket)

    #
    # Define HeapStorageInterface Methods
    #

    def clone_device(self):
        return HeapStorage(self._storage.clone_device())

    @classmethod
    def compute_storage_size(cls,
                             block_size,
                             heap_height,
                             blocks_per_bucket=1,
                             heap_base=2,
                             ignore_header=False,
                             storage_type='file',
                             **kwds):
        assert (block_size > 0) and (block_size == int(block_size))
        assert heap_height >= 0
        assert blocks_per_bucket >= 1
        assert heap_base >= 2
        assert 'block_count' not in kwds
        vheap = SizedVirtualHeap(
            heap_base,
            heap_height,
            blocks_per_bucket=blocks_per_bucket)
        if ignore_header:
            return BlockStorageTypeFactory(storage_type).\
                compute_storage_size(
                    vheap.blocks_per_bucket * block_size,
                    vheap.bucket_count(),
                    ignore_header=True,
                    **kwds)
        else:
            return cls._header_offset + \
                BlockStorageTypeFactory(storage_type).\
                compute_storage_size(
                    vheap.blocks_per_bucket * block_size,
                    vheap.bucket_count(),
                    ignore_header=False,
                    **kwds)

    @classmethod
    def setup(cls,
              storage_name,
              block_size,
              heap_height,
              blocks_per_bucket=1,
              heap_base=2,
              storage_type='file',
              **kwds):
        if 'block_count' in kwds:
            raise ValueError("'block_count' keyword is not accepted")
        if heap_height < 0:
            raise ValueError(
                "heap height must be 0 or greater. Invalid value: %s"
                % (heap_height))
        if blocks_per_bucket < 1:
            raise ValueError(
                "blocks_per_bucket must be 1 or greater. "
                "Invalid value: %s" % (blocks_per_bucket))
        if heap_base < 2:
            raise ValueError(
                "heap base must be 2 or greater. Invalid value: %s"
                % (heap_base))

        vheap = SizedVirtualHeap(
            heap_base,
            heap_height,
            blocks_per_bucket=blocks_per_bucket)

        user_header_data = kwds.pop('header_data', bytes())
        if type(user_header_data) is not bytes:
            raise TypeError(
                "'header_data' must be of type bytes. "
                "Invalid type: %s" % (type(user_header_data)))
        kwds['header_data'] = \
            struct.pack(cls._header_struct_string,
                        heap_base,
                        heap_height,
                        blocks_per_bucket) + \
            user_header_data

        return HeapStorage(
            BlockStorageTypeFactory(storage_type).setup(
                storage_name,
                vheap.blocks_per_bucket * block_size,
                vheap.bucket_count(),
                **kwds))

    @property
    def header_data(self):
        return self._storage.header_data[self._header_offset:]

    @property
    def bucket_count(self):
        return self._storage.block_count

    @property
    def bucket_size(self):
        return self._storage.block_size

    @property
    def blocks_per_bucket(self):
        return self._vheap.blocks_per_bucket

    @property
    def storage_name(self):
        return self._storage.storage_name

    @property
    def virtual_heap(self):
        return self._vheap

    @property
    def bucket_storage(self):
        return self._storage

    def update_header_data(self, new_header_data):
        self._storage.update_header_data(
            self._storage.header_data[:self._header_offset] + \
            new_header_data)

    def close(self):
        self._storage.close()

    def read_path(self, b, level_start=0):
        assert 0 <= b < self._vheap.bucket_count()
        bucket_list = self._vheap.Node(b).bucket_path_from_root()
        assert 0 <= level_start < len(bucket_list)
        return self._storage.read_blocks(bucket_list[level_start:])

    def write_path(self, b, buckets, level_start=0):
        assert 0 <= b < self._vheap.bucket_count()
        bucket_list = self._vheap.Node(b).bucket_path_from_root()
        assert 0 <= level_start < len(bucket_list)
        self._storage.write_blocks(bucket_list[level_start:],
                                   buckets)

    @property
    def bytes_sent(self):
        return self._storage.bytes_sent

    @property
    def bytes_received(self):
        return self._storage.bytes_received
