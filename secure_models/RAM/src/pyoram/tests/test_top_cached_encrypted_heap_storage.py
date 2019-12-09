import os
import unittest
import tempfile
import random

from pyoram.storage.block_storage import \
    BlockStorageTypeFactory
from pyoram.encrypted_storage.top_cached_encrypted_heap_storage import \
    TopCachedEncryptedHeapStorage
from pyoram.encrypted_storage.encrypted_block_storage import \
    EncryptedBlockStorage
from pyoram.encrypted_storage.encrypted_heap_storage import \
    EncryptedHeapStorage
from pyoram.crypto.aes import AES

from six.moves import xrange

thisdir = os.path.dirname(os.path.abspath(__file__))

class _TestTopCachedEncryptedHeapStorage(object):

    _init_kwds = None
    _storage_type = None
    _heap_base = None
    _heap_height = None

    @classmethod
    def setUpClass(cls):
        assert cls._init_kwds is not None
        assert cls._storage_type is not None
        assert cls._heap_base is not None
        assert cls._heap_height is not None
        fd, cls._dummy_name = tempfile.mkstemp()
        os.close(fd)
        try:
            os.remove(cls._dummy_name)
        except OSError:                                # pragma: no cover
            pass                                       # pragma: no cover
        cls._block_size = 50
        cls._blocks_per_bucket = 3
        cls._bucket_count = \
            ((cls._heap_base**(cls._heap_height+1)) - 1)//(cls._heap_base-1)
        cls._block_count = cls._bucket_count * \
                           cls._blocks_per_bucket
        cls._testfname = cls.__name__ + "_testfile.bin"
        cls._buckets = []
        f = EncryptedHeapStorage.setup(
            cls._testfname,
            cls._block_size,
            cls._heap_height,
            heap_base=cls._heap_base,
            blocks_per_bucket=cls._blocks_per_bucket,
            storage_type=cls._storage_type,
            initialize=lambda i: bytes(bytearray([i]) * \
                                       cls._block_size * \
                                       cls._blocks_per_bucket),
            ignore_existing=True)
        f.close()
        cls._key = f.key
        for i in range(cls._bucket_count):
            data = bytearray([i]) * \
                   cls._block_size * \
                   cls._blocks_per_bucket
            cls._buckets.append(data)

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls._testfname)
        except OSError:                                # pragma: no cover
            pass                                       # pragma: no cover
        try:
            os.remove(cls._dummy_name)
        except OSError:                                # pragma: no cover
            pass                                       # pragma: no cover

    def test_factory(self):
        kwds = dict(self._init_kwds)
        kwds['cached_levels'] = 0
        with EncryptedHeapStorage(
                self._testfname,
                key=self._key,
                storage_type=self._storage_type) as f1:
            with TopCachedEncryptedHeapStorage(f1, **kwds) as f2:
                self.assertTrue(f1 is f2)

    def test_setup(self):
        fname = ".".join(self.id().split(".")[1:])
        fname += ".bin"
        fname = os.path.join(thisdir, fname)
        if os.path.exists(fname):
            os.remove(fname)                           # pragma: no cover
        bsize = 10
        blocks_per_bucket = 3
        fsetup = EncryptedHeapStorage.setup(
            fname,
            bsize,
            self._heap_height,
            heap_base=self._heap_base,
            storage_type=self._storage_type,
            blocks_per_bucket=blocks_per_bucket)
        fsetup.close()
        self.assertEqual(type(fsetup.raw_storage),
                         BlockStorageTypeFactory(self._storage_type))
        with open(fname, 'rb') as f:
            flen = len(f.read())
            self.assertEqual(
                flen,
                TopCachedEncryptedHeapStorage.compute_storage_size(
                    bsize,
                    self._heap_height,
                    heap_base=self._heap_base,
                    blocks_per_bucket=blocks_per_bucket))
            self.assertEqual(
                flen >
                TopCachedEncryptedHeapStorage.compute_storage_size(
                    bsize,
                    self._heap_height,
                    heap_base=self._heap_base,
                    blocks_per_bucket=blocks_per_bucket,
                    ignore_header=True),
                True)
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(
                    fname,
                    key=fsetup.key,
                    storage_type=self._storage_type),
                **self._init_kwds) as f:
            self.assertEqual(f.header_data, bytes())
            self.assertEqual(fsetup.header_data, bytes())
            self.assertEqual(f.key, fsetup.key)
            self.assertEqual(f.blocks_per_bucket,
                             blocks_per_bucket)
            self.assertEqual(fsetup.blocks_per_bucket,
                             blocks_per_bucket)
            self.assertEqual(f.bucket_count,
                             (self._heap_base**(self._heap_height+1) - 1)//(self._heap_base-1))
            self.assertEqual(fsetup.bucket_count,
                             (self._heap_base**(self._heap_height+1) - 1)//(self._heap_base-1))
            self.assertEqual(f.bucket_size,
                             bsize * blocks_per_bucket)
            self.assertEqual(fsetup.bucket_size,
                             bsize * blocks_per_bucket)
            self.assertEqual(f.storage_name, fname)
            self.assertEqual(fsetup.storage_name, fname)
        os.remove(fname)

    def test_setup_withdata(self):
        fname = ".".join(self.id().split(".")[1:])
        fname += ".bin"
        fname = os.path.join(thisdir, fname)
        if os.path.exists(fname):
            os.remove(fname)                           # pragma: no cover
        bsize = 10
        blocks_per_bucket = 1
        header_data = bytes(bytearray([0,1,2]))
        fsetup = EncryptedHeapStorage.setup(
            fname,
            bsize,
            self._heap_height,
            heap_base=self._heap_base,
            storage_type=self._storage_type,
            blocks_per_bucket=blocks_per_bucket,
            header_data=header_data)
        fsetup.close()
        self.assertEqual(type(fsetup.raw_storage),
                         BlockStorageTypeFactory(self._storage_type))
        with open(fname, 'rb') as f:
            flen = len(f.read())
            self.assertEqual(
                flen,
                TopCachedEncryptedHeapStorage.compute_storage_size(
                    bsize,
                    self._heap_height,
                    heap_base=self._heap_base,
                    header_data=header_data))
            self.assertTrue(len(header_data) > 0)
            self.assertEqual(
                TopCachedEncryptedHeapStorage.compute_storage_size(
                    bsize,
                    self._heap_height,
                    heap_base=self._heap_base,
                    storage_type=self._storage_type) <
                TopCachedEncryptedHeapStorage.compute_storage_size(
                    bsize,
                    self._heap_height,
                    heap_base=self._heap_base,
                    storage_type=self._storage_type,
                    header_data=header_data),
                True)
            self.assertEqual(
                flen >
                TopCachedEncryptedHeapStorage.compute_storage_size(
                    bsize,
                    self._heap_height,
                    heap_base=self._heap_base,
                    storage_type=self._storage_type,
                    header_data=header_data,
                    ignore_header=True),
                True)
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(
                    fname,
                    key=fsetup.key,
                    storage_type=self._storage_type),
                **self._init_kwds) as f:
            self.assertEqual(f.header_data, header_data)
            self.assertEqual(fsetup.header_data, header_data)
            self.assertEqual(f.key, fsetup.key)
            self.assertEqual(f.blocks_per_bucket,
                             blocks_per_bucket)
            self.assertEqual(fsetup.blocks_per_bucket,
                             blocks_per_bucket)
            self.assertEqual(f.bucket_count,
                             (self._heap_base**(self._heap_height+1) - 1)//(self._heap_base-1))
            self.assertEqual(fsetup.bucket_count,
                             (self._heap_base**(self._heap_height+1) - 1)//(self._heap_base-1))
            self.assertEqual(f.bucket_size,
                             bsize * blocks_per_bucket)
            self.assertEqual(fsetup.bucket_size,
                             bsize * blocks_per_bucket)
            self.assertEqual(f.storage_name, fname)
            self.assertEqual(fsetup.storage_name, fname)
        os.remove(fname)

    def test_init_exists(self):
        self.assertEqual(os.path.exists(self._testfname), True)
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._storage_type) as f:
            databefore = f.read_blocks(list(range(f.block_count)))
        with self.assertRaises(ValueError):
            with EncryptedBlockStorage(self._testfname,
                                       key=self._key,
                                       storage_type=self._storage_type) as fb:
                with TopCachedEncryptedHeapStorage(
                        EncryptedHeapStorage(fb, key=self._key),
                        **self._init_kwds) as f:
                    pass                               # pragma: no cover
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(
                    self._testfname,
                    key=self._key,
                    storage_type=self._storage_type),
                **self._init_kwds) as f:
            self.assertEqual(f.key, self._key)
            self.assertEqual(f.bucket_size,
                             self._block_size * \
                             self._blocks_per_bucket)
            self.assertEqual(f.bucket_count,
                             self._bucket_count)
            self.assertEqual(f.storage_name, self._testfname)
            self.assertEqual(f.header_data, bytes())
        self.assertEqual(os.path.exists(self._testfname), True)
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(
                    self._testfname,
                    key=self._key,
                    storage_type=self._storage_type),
                **self._init_kwds) as f:
            dataafter = f.bucket_storage.read_blocks(
                list(range(f.bucket_storage.block_count)))
        self.assertEqual(databefore, dataafter)

    def test_read_path(self):
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(
                    self._testfname,
                    key=self._key,
                    storage_type=self._storage_type),
                **self._init_kwds) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

            self.assertEqual(
                f.virtual_heap.first_bucket_at_level(0), 0)
            self.assertNotEqual(
                f.virtual_heap.last_leaf_bucket(), 0)
            total_buckets = 0
            for b in range(f.virtual_heap.first_bucket_at_level(0),
                           f.virtual_heap.last_leaf_bucket()+1):
                full_bucket_path = f.virtual_heap.Node(b).\
                                   bucket_path_from_root()
                all_level_starts = list(range(len(full_bucket_path)+1))
                for level_start in all_level_starts:
                    data = f.read_path(b, level_start=level_start)
                    bucket_path = full_bucket_path[level_start:]

                    if len(full_bucket_path) <= f._external_level:
                        pass
                    elif level_start >= f._external_level:
                        total_buckets += len(bucket_path)
                    else:
                        total_buckets += len(full_bucket_path[f._external_level:])

                    self.assertEqual(f.virtual_heap.Node(b).level+1-level_start,
                                     len(bucket_path))
                    for i, bucket in zip(bucket_path, data):
                        self.assertEqual(list(bytearray(bucket)),
                                         list(self._buckets[i]))

            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received,
                             total_buckets*f.bucket_storage._storage.block_size)

    def test_write_path(self):
        data = [bytearray([self._bucket_count]) * \
                self._block_size * \
                self._blocks_per_bucket
                for i in xrange(self._block_count)]
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(
                    self._testfname,
                    key=self._key,
                    storage_type=self._storage_type),
                **self._init_kwds) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

            self.assertEqual(
                f.virtual_heap.first_bucket_at_level(0), 0)
            self.assertNotEqual(
                f.virtual_heap.last_leaf_bucket(), 0)
            all_buckets = list(range(f.virtual_heap.first_bucket_at_level(0),
                                     f.virtual_heap.last_leaf_bucket()+1))
            random.shuffle(all_buckets)
            total_read_buckets = 0
            total_write_buckets = 0
            for b in all_buckets:
                full_bucket_path = f.virtual_heap.Node(b).\
                                   bucket_path_from_root()
                all_level_starts = list(range(len(full_bucket_path)+1))
                random.shuffle(all_level_starts)
                for level_start in all_level_starts:
                    orig = f.read_path(b, level_start=level_start)
                    bucket_path = full_bucket_path[level_start:]

                    if len(full_bucket_path) <= f._external_level:
                        pass
                    elif level_start >= f._external_level:
                        total_read_buckets += len(bucket_path)
                    else:
                        total_read_buckets += len(full_bucket_path[f._external_level:])

                    if level_start != len(full_bucket_path):
                        self.assertNotEqual(len(bucket_path), 0)
                    self.assertEqual(f.virtual_heap.Node(b).level+1-level_start,
                                     len(bucket_path))
                    self.assertEqual(len(orig), len(bucket_path))

                    for i, bucket in zip(bucket_path, orig):
                        self.assertEqual(list(bytearray(bucket)),
                                         list(self._buckets[i]))

                    f.write_path(b, [bytes(data[i])
                                     for i in bucket_path],
                                 level_start=level_start)
                    if len(full_bucket_path) <= f._external_level:
                        pass
                    elif level_start >= f._external_level:
                        total_write_buckets += len(bucket_path)
                    else:
                        total_write_buckets += len(full_bucket_path[f._external_level:])

                    new = f.read_path(b, level_start=level_start)
                    if len(full_bucket_path) <= f._external_level:
                        pass
                    elif level_start >= f._external_level:
                        total_read_buckets += len(bucket_path)
                    else:
                        total_read_buckets += len(full_bucket_path[f._external_level:])

                    self.assertEqual(len(new), len(bucket_path))
                    for i, bucket in zip(bucket_path, new):
                        self.assertEqual(list(bytearray(bucket)),
                                         list(data[i]))

                    f.write_path(b, [bytes(self._buckets[i])
                                     for i in bucket_path],
                                 level_start=level_start)
                    if len(full_bucket_path) <= f._external_level:
                        pass
                    elif level_start >= f._external_level:
                        total_write_buckets += len(bucket_path)
                    else:
                        total_write_buckets += len(full_bucket_path[f._external_level:])


                    orig = f.read_path(b, level_start=level_start)
                    if len(full_bucket_path) <= f._external_level:
                        pass
                    elif level_start >= f._external_level:
                        total_read_buckets += len(bucket_path)
                    else:
                        total_read_buckets += len(full_bucket_path[f._external_level:])

                    self.assertEqual(len(orig), len(bucket_path))
                    for i, bucket in zip(bucket_path, orig):
                        self.assertEqual(list(bytearray(bucket)),
                                         list(self._buckets[i]))

                    full_orig = f.read_path(b)
                    if len(full_bucket_path) <= f._external_level:
                        pass
                    else:
                        total_read_buckets += len(full_bucket_path[f._external_level:])

                    for i, bucket in zip(full_bucket_path, full_orig):
                        self.assertEqual(list(bytearray(bucket)),
                                         list(self._buckets[i]))
                    for c in xrange(self._heap_base):
                        cn = f.virtual_heap.Node(b).child_node(c)
                        if not f.virtual_heap.is_nil_node(cn):
                            cb = cn.bucket
                            bucket_path = f.virtual_heap.Node(cb).\
                                          bucket_path_from_root()
                            orig = f.read_path(cb)
                            if len(bucket_path) <= f._external_level:
                                pass
                            else:
                                total_read_buckets += len(bucket_path[f._external_level:])
                            self.assertEqual(len(orig), len(bucket_path))
                            for i, bucket in zip(bucket_path, orig):
                                self.assertEqual(list(bytearray(bucket)),
                                                 list(self._buckets[i]))

            self.assertEqual(f.bytes_sent,
                             total_write_buckets*f.bucket_storage._storage.block_size)
            self.assertEqual(f.bytes_received,
                             total_read_buckets*f.bucket_storage._storage.block_size)

    def test_update_header_data(self):
        fname = ".".join(self.id().split(".")[1:])
        fname += ".bin"
        fname = os.path.join(thisdir, fname)
        if os.path.exists(fname):
            os.remove(fname)                           # pragma: no cover
        bsize = 10
        blocks_per_bucket = 1
        header_data = bytes(bytearray([0,1,2]))
        fsetup = EncryptedHeapStorage.setup(
            fname,
            bsize,
            self._heap_height,
            heap_base=self._heap_base,
            blocks_per_bucket=blocks_per_bucket,
            header_data=header_data)
        fsetup.close()
        new_header_data = bytes(bytearray([1,1,1]))
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(
                    fname,
                    key=fsetup.key,
                    storage_type=self._storage_type),
                **self._init_kwds) as f:
            self.assertEqual(f.header_data, header_data)
            f.update_header_data(new_header_data)
            self.assertEqual(f.header_data, new_header_data)
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(
                    fname,
                    key=fsetup.key,
                    storage_type=self._storage_type),
                **self._init_kwds) as f:
            self.assertEqual(f.header_data, new_header_data)
        with self.assertRaises(ValueError):
            with TopCachedEncryptedHeapStorage(
                    EncryptedHeapStorage(
                        fname,
                        key=fsetup.key,
                        storage_type=self._storage_type),
                    **self._init_kwds) as f:
                f.update_header_data(bytes(bytearray([1,1])))
        with self.assertRaises(ValueError):
            with TopCachedEncryptedHeapStorage(
                    EncryptedHeapStorage(
                        fname,
                        key=fsetup.key,
                        storage_type=self._storage_type),
                    **self._init_kwds) as f:
                f.update_header_data(bytes(bytearray([1,1,1,1])))
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(
                    fname,
                    key=fsetup.key,
                    storage_type=self._storage_type),
                **self._init_kwds) as f:
            self.assertEqual(f.header_data, new_header_data)
        os.remove(fname)

    def test_locked_flag(self):
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(self._testfname,
                                     key=self._key,
                                     storage_type=self._storage_type),
                **self._init_kwds) as f:
            with self.assertRaises(IOError):
                with TopCachedEncryptedHeapStorage(
                        EncryptedHeapStorage(self._testfname,
                                             key=self._key,
                                             storage_type=self._storage_type),
                        **self._init_kwds) as f1:
                    pass                               # pragma: no cover
            with self.assertRaises(IOError):
                with TopCachedEncryptedHeapStorage(
                        EncryptedHeapStorage(self._testfname,
                                             key=self._key,
                                             storage_type=self._storage_type),
                        **self._init_kwds) as f1:
                    pass                               # pragma: no cover
            with TopCachedEncryptedHeapStorage(
                    EncryptedHeapStorage(self._testfname,
                                         key=self._key,
                                         storage_type=self._storage_type,
                                         ignore_lock=True),
                    **self._init_kwds) as f1:
                pass
            with self.assertRaises(IOError):
                with TopCachedEncryptedHeapStorage(
                        EncryptedHeapStorage(self._testfname,
                                             key=self._key,
                                             storage_type=self._storage_type),
                        **self._init_kwds) as f1:
                    pass                               # pragma: no cover
            with TopCachedEncryptedHeapStorage(
                    EncryptedHeapStorage(self._testfname,
                                         key=self._key,
                                         storage_type=self._storage_type,
                                         ignore_lock=True),
                    **self._init_kwds) as f1:
                pass
            with TopCachedEncryptedHeapStorage(
                    EncryptedHeapStorage(self._testfname,
                                         key=self._key,
                                         storage_type=self._storage_type,
                                         ignore_lock=True),
                    **self._init_kwds) as f1:
                pass
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(self._testfname,
                                     key=self._key,
                                     storage_type=self._storage_type),
                **self._init_kwds) as f:
            pass

    def test_cache_size(self):
        with TopCachedEncryptedHeapStorage(
                EncryptedHeapStorage(self._testfname,
                                     key=self._key,
                                     storage_type=self._storage_type),
                **self._init_kwds) as f:
            num_cached_levels = self._init_kwds.get('cached_levels', 1)
            if num_cached_levels < 0:
                num_cached_levels = f.virtual_heap.levels
            cache_bucket_count = 0
            for l in xrange(num_cached_levels):
                if l <= f.virtual_heap.last_level:
                    cache_bucket_count += f.virtual_heap.bucket_count_at_level(l)
            self.assertEqual(cache_bucket_count > 0, True)
            self.assertEqual(len(f.cached_bucket_data),
                             cache_bucket_count * f.bucket_size)

            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)
            self.assertEqual(f._root_device.bytes_sent, 0)
            self.assertEqual(
                f._root_device.bytes_received,
                cache_bucket_count*f._root_device.bucket_storage._storage.block_size)

class TestTopCachedEncryptedHeapStorageCacheMMapDefault(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {}
    _storage_type = 'mmap'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageMMapCache1(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 1}
    _storage_type = 'mmap'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageMMapCache2(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 2}
    _storage_type = 'mmap'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageMMapCache3(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 3}
    _storage_type = 'mmap'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageMMapCache4(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 4}
    _storage_type = 'mmap'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageMMapCache5(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 5}
    _storage_type = 'mmap'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageCacheFileDefault(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageFileCache1(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 1}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageFileCache2(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 2}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageFileCache3(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 3}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageFileCache4(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 4}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageFileCache5(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 5}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageFileCacheBigConcurrency0(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 20,
                  'concurrency_level': 0}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageFileCache6Concurrency1(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 6,
                  'concurrency_level': 1}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageFileCache3ConcurrencyBig(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 3,
                  'concurrency_level': 20}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 7

class TestTopCachedEncryptedHeapStorageFileCache3Concurrency1Base3(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': 3,
                  'concurrency_level': 3}
    _storage_type = 'file'
    _heap_base = 3
    _heap_height = 4

class TestTopCachedEncryptedHeapStorageFileCacheAll(
        _TestTopCachedEncryptedHeapStorage,
        unittest.TestCase):
    _init_kwds = {'cached_levels': -1}
    _storage_type = 'file'
    _heap_base = 2
    _heap_height = 3

if __name__ == "__main__":
    unittest.main()                                    # pragma: no cover
