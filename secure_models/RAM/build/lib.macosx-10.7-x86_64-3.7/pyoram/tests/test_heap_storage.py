import os
import unittest
import tempfile

from pyoram.util.virtual_heap import \
    SizedVirtualHeap
from pyoram.storage.block_storage import \
    BlockStorageTypeFactory
from pyoram.storage.block_storage_file import \
    BlockStorageFile
from pyoram.storage.heap_storage import \
    HeapStorage

from six.moves import xrange

thisdir = os.path.dirname(os.path.abspath(__file__))

class TestHeapStorage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fd, cls._dummy_name = tempfile.mkstemp()
        os.close(fd)
        try:
            os.remove(cls._dummy_name)
        except OSError:                                # pragma: no cover
            pass                                       # pragma: no cover
        cls._block_size = 25
        cls._blocks_per_bucket = 3
        cls._heap_base = 4
        cls._heap_height = 2
        cls._bucket_count = \
            ((cls._heap_base**(cls._heap_height+1)) - 1)//(cls._heap_base-1)
        cls._block_count = cls._bucket_count * \
                           cls._blocks_per_bucket
        cls._testfname = cls.__name__ + "_testfile.bin"
        cls._buckets = []
        cls._type_name = "file"
        f = HeapStorage.setup(
            cls._testfname,
            block_size=cls._block_size,
            heap_height=cls._heap_height,
            heap_base=cls._heap_base,
            blocks_per_bucket=cls._blocks_per_bucket,
            storage_type=cls._type_name,
            initialize=lambda i: bytes(bytearray([i]) * \
                                       cls._block_size * \
                                       cls._blocks_per_bucket),
            ignore_existing=True)
        f.close()
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

    def test_setup_fails(self):
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(IOError):
            HeapStorage.setup(
                os.path.join(thisdir,
                             "baselines",
                             "exists.empty"),
                block_size=10,
                heap_height=1,
                blocks_per_bucket=1,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(IOError):
            HeapStorage.setup(
                os.path.join(thisdir,
                             "baselines",
                             "exists.empty"),
                block_size=10,
                heap_height=1,
                blocks_per_bucket=1,
                storage_type=self._type_name,
                ignore_existing=False)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        # bad block_size
        with self.assertRaises(ValueError):
            HeapStorage.setup(
                self._dummy_name,
                block_size=0,
                heap_height=1,
                blocks_per_bucket=1,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        # bad heap_height
        with self.assertRaises(ValueError):
            HeapStorage.setup(
                self._dummy_name,
                block_size=1,
                heap_height=-1,
                blocks_per_bucket=1,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        # bad blocks_per_bucket
        with self.assertRaises(ValueError):
            HeapStorage.setup(
                self._dummy_name,
                block_size=1,
                heap_height=1,
                blocks_per_bucket=0,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        # bad heap_base
        with self.assertRaises(ValueError):
            HeapStorage.setup(
                self._dummy_name,
                block_size=1,
                heap_height=1,
                blocks_per_bucket=1,
                heap_base=1,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        # bad header_data
        with self.assertRaises(TypeError):
            HeapStorage.setup(
                self._dummy_name,
                block_size=1,
                heap_height=1,
                blocks_per_bucket=1,
                storage_type=self._type_name,
                header_data=2)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        # uses block_count
        with self.assertRaises(ValueError):
            HeapStorage.setup(
                self._dummy_name,
                block_size=1,
                heap_height=1,
                blocks_per_bucket=1,
                block_count=1,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)

    def test_setup(self):
        fname = ".".join(self.id().split(".")[1:])
        fname += ".bin"
        fname = os.path.join(thisdir, fname)
        if os.path.exists(fname):
            os.remove(fname)                           # pragma: no cover
        bsize = 10
        heap_height = 2
        blocks_per_bucket = 3
        fsetup = HeapStorage.setup(
            fname,
            bsize,
            heap_height,
            blocks_per_bucket=blocks_per_bucket)
        fsetup.close()
        self.assertEqual(type(fsetup.bucket_storage),
                         BlockStorageTypeFactory(self._type_name))
        with open(fname, 'rb') as f:
            flen = len(f.read())
            self.assertEqual(
                flen,
                HeapStorage.compute_storage_size(
                    bsize,
                    heap_height,
                    blocks_per_bucket=blocks_per_bucket))
            self.assertEqual(
                flen >
                HeapStorage.compute_storage_size(
                    bsize,
                    heap_height,
                    blocks_per_bucket=blocks_per_bucket,
                    ignore_header=True),
                True)
        with HeapStorage(
                fname,
                storage_type=self._type_name) as f:
            self.assertEqual(f.header_data, bytes())
            self.assertEqual(fsetup.header_data, bytes())
            self.assertEqual(f.blocks_per_bucket,
                             blocks_per_bucket)
            self.assertEqual(fsetup.blocks_per_bucket,
                             blocks_per_bucket)
            self.assertEqual(f.bucket_count,
                             2**(heap_height+1) - 1)
            self.assertEqual(fsetup.bucket_count,
                             2**(heap_height+1) - 1)
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
        heap_height = 2
        blocks_per_bucket = 1
        header_data = bytes(bytearray([0,1,2]))
        fsetup = HeapStorage.setup(
            fname,
            bsize,
            heap_height,
            blocks_per_bucket=blocks_per_bucket,
            header_data=header_data)
        fsetup.close()
        self.assertEqual(type(fsetup.bucket_storage),
                         BlockStorageTypeFactory(self._type_name))
        with open(fname, 'rb') as f:
            flen = len(f.read())
            self.assertEqual(
                flen,
                HeapStorage.compute_storage_size(
                    bsize,
                    heap_height,
                    header_data=header_data))
            self.assertTrue(len(header_data) > 0)
            self.assertEqual(
                HeapStorage.compute_storage_size(
                    bsize,
                    heap_height,
                    storage_type=self._type_name) <
                HeapStorage.compute_storage_size(
                    bsize,
                    heap_height,
                    storage_type=self._type_name,
                    header_data=header_data),
                True)
            self.assertEqual(
                flen >
                HeapStorage.compute_storage_size(
                    bsize,
                    heap_height,
                    storage_type=self._type_name,
                    header_data=header_data,
                    ignore_header=True),
                True)
        with HeapStorage(
                fname,
                storage_type=self._type_name) as f:
            self.assertEqual(f.header_data, header_data)
            self.assertEqual(fsetup.header_data, header_data)
            self.assertEqual(f.blocks_per_bucket,
                             blocks_per_bucket)
            self.assertEqual(fsetup.blocks_per_bucket,
                             blocks_per_bucket)
            self.assertEqual(f.bucket_count,
                             2**(heap_height+1) - 1)
            self.assertEqual(fsetup.bucket_count,
                             2**(heap_height+1) - 1)
            self.assertEqual(f.bucket_size,
                             bsize * blocks_per_bucket)
            self.assertEqual(fsetup.bucket_size,
                             bsize * blocks_per_bucket)
            self.assertEqual(f.storage_name, fname)
            self.assertEqual(fsetup.storage_name, fname)
        os.remove(fname)

    def test_init_noexists(self):
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(IOError):
            with HeapStorage(
                    self._dummy_name,
                    storage_type=self._type_name) as f:
                pass                                   # pragma: no cover

    def test_init_exists(self):
        self.assertEqual(os.path.exists(self._testfname), True)
        with open(self._testfname, 'rb') as f:
            databefore = f.read()
        with self.assertRaises(ValueError):
            with BlockStorageFile(self._testfname) as fb:
                with HeapStorage(fb, storage_type='file') as f:
                    pass                               # pragma: no cover
        with HeapStorage(
                self._testfname,
                storage_type=self._type_name) as f:
            self.assertEqual(f.bucket_size,
                             self._block_size * \
                             self._blocks_per_bucket)
            self.assertEqual(f.bucket_count,
                             self._bucket_count)
            self.assertEqual(f.storage_name, self._testfname)
            self.assertEqual(f.header_data, bytes())
        self.assertEqual(os.path.exists(self._testfname), True)
        with open(self._testfname, 'rb') as f:
            dataafter = f.read()
        self.assertEqual(databefore, dataafter)

    def test_read_path(self):

        with HeapStorage(
                self._testfname,
                storage_type=self._type_name) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

            self.assertEqual(
                f.virtual_heap.first_bucket_at_level(0), 0)
            self.assertNotEqual(
                f.virtual_heap.last_leaf_bucket(), 0)
            total_buckets = 0
            for b in range(f.virtual_heap.first_bucket_at_level(0),
                           f.virtual_heap.last_leaf_bucket()+1):
                data = f.read_path(b)
                bucket_path = f.virtual_heap.Node(b).\
                              bucket_path_from_root()
                total_buckets += len(bucket_path)
                self.assertEqual(f.virtual_heap.Node(b).level+1,
                                 len(bucket_path))
                for i, bucket in zip(bucket_path, data):
                    self.assertEqual(list(bytearray(bucket)),
                                     list(self._buckets[i]))

            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received,
                             total_buckets*f.bucket_storage.block_size)

    def test_write_path(self):
        data = [bytearray([self._bucket_count]) * \
                self._block_size * \
                self._blocks_per_bucket
                for i in xrange(self._block_count)]
        with HeapStorage(
                self._testfname,
                storage_type=self._type_name) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

            self.assertEqual(
                f.virtual_heap.first_bucket_at_level(0), 0)
            self.assertNotEqual(
                f.virtual_heap.last_leaf_bucket(), 0)
            total_buckets = 0
            for b in range(f.virtual_heap.first_bucket_at_level(0),
                           f.virtual_heap.last_leaf_bucket()+1):
                orig = f.read_path(b)
                bucket_path = f.virtual_heap.Node(b).\
                              bucket_path_from_root()
                total_buckets += len(bucket_path)
                self.assertNotEqual(len(bucket_path), 0)
                self.assertEqual(f.virtual_heap.Node(b).level+1,
                                 len(bucket_path))
                self.assertEqual(len(orig), len(bucket_path))
                for i, bucket in zip(bucket_path, orig):
                    self.assertEqual(list(bytearray(bucket)),
                                     list(self._buckets[i]))
                f.write_path(b, [bytes(data[i])
                                 for i in bucket_path])

                new = f.read_path(b)
                self.assertEqual(len(new), len(bucket_path))
                for i, bucket in zip(bucket_path, new):
                    self.assertEqual(list(bytearray(bucket)),
                                     list(data[i]))

                f.write_path(b, [bytes(self._buckets[i])
                                 for i in bucket_path])

                orig = f.read_path(b)
                self.assertEqual(len(orig), len(bucket_path))
                for i, bucket in zip(bucket_path, orig):
                    self.assertEqual(list(bytearray(bucket)),
                                     list(self._buckets[i]))

            self.assertEqual(f.bytes_sent,
                             total_buckets*f.bucket_storage.block_size*2)
            self.assertEqual(f.bytes_received,
                             total_buckets*f.bucket_storage.block_size*3)

    def test_update_header_data(self):
        fname = ".".join(self.id().split(".")[1:])
        fname += ".bin"
        fname = os.path.join(thisdir, fname)
        if os.path.exists(fname):
            os.remove(fname)                           # pragma: no cover
        bsize = 10
        heap_height = 2
        blocks_per_bucket = 1
        header_data = bytes(bytearray([0,1,2]))
        fsetup = HeapStorage.setup(
            fname,
            block_size=bsize,
            heap_height=heap_height,
            blocks_per_bucket=blocks_per_bucket,
            header_data=header_data)
        fsetup.close()
        new_header_data = bytes(bytearray([1,1,1]))
        with HeapStorage(
                fname,
                storage_type=self._type_name) as f:
            self.assertEqual(f.header_data, header_data)
            f.update_header_data(new_header_data)
            self.assertEqual(f.header_data, new_header_data)
        with HeapStorage(
                fname,
                storage_type=self._type_name) as f:
            self.assertEqual(f.header_data, new_header_data)
        with self.assertRaises(ValueError):
            with HeapStorage(
                    fname,
                    storage_type=self._type_name) as f:
                f.update_header_data(bytes(bytearray([1,1])))
        with self.assertRaises(ValueError):
            with HeapStorage(
                    fname,
                    storage_type=self._type_name) as f:
                f.update_header_data(bytes(bytearray([1,1,1,1])))
        with HeapStorage(
                fname,
                storage_type=self._type_name) as f:
            self.assertEqual(f.header_data, new_header_data)
        os.remove(fname)

    def test_locked_flag(self):
        with HeapStorage(self._testfname,
                                  storage_type=self._type_name) as f:
            with self.assertRaises(IOError):
                with HeapStorage(self._testfname,
                                          storage_type=self._type_name) as f1:
                    pass                               # pragma: no cover
            with self.assertRaises(IOError):
                with HeapStorage(self._testfname,
                                          storage_type=self._type_name) as f1:
                    pass                               # pragma: no cover
            with HeapStorage(self._testfname,
                                      storage_type=self._type_name,
                                      ignore_lock=True) as f1:
                pass
            with self.assertRaises(IOError):
                with HeapStorage(self._testfname,
                                          storage_type=self._type_name) as f1:
                    pass                               # pragma: no cover
            with HeapStorage(self._testfname,
                                      storage_type=self._type_name,
                                      ignore_lock=True) as f1:
                pass
            with HeapStorage(self._testfname,
                                      storage_type=self._type_name,
                                      ignore_lock=True) as f1:
                pass
        with HeapStorage(self._testfname,
                                  storage_type=self._type_name) as f:
            pass

    def test_read_path_cloned(self):

        with HeapStorage(
                self._testfname,
                storage_type=self._type_name) as forig:
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)
            with forig.clone_device() as f:
                self.assertEqual(forig.bytes_sent, 0)
                self.assertEqual(forig.bytes_received, 0)
                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received, 0)

                self.assertEqual(
                    f.virtual_heap.first_bucket_at_level(0), 0)
                self.assertNotEqual(
                    f.virtual_heap.last_leaf_bucket(), 0)
                total_buckets = 0
                for b in range(f.virtual_heap.first_bucket_at_level(0),
                               f.virtual_heap.last_leaf_bucket()+1):
                    data = f.read_path(b)
                    bucket_path = f.virtual_heap.Node(b).\
                                  bucket_path_from_root()
                    total_buckets += len(bucket_path)
                    self.assertEqual(f.virtual_heap.Node(b).level+1,
                                     len(bucket_path))
                    for i, bucket in zip(bucket_path, data):
                        self.assertEqual(list(bytearray(bucket)),
                                         list(self._buckets[i]))

                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received,
                                 total_buckets*f.bucket_storage.block_size)
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)

    def test_write_path_cloned(self):
        data = [bytearray([self._bucket_count]) * \
                self._block_size * \
                self._blocks_per_bucket
                for i in xrange(self._block_count)]
        with HeapStorage(
                self._testfname,
                storage_type=self._type_name) as forig:
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)
            with forig.clone_device() as f:
                self.assertEqual(forig.bytes_sent, 0)
                self.assertEqual(forig.bytes_received, 0)
                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received, 0)

                self.assertEqual(
                    f.virtual_heap.first_bucket_at_level(0), 0)
                self.assertNotEqual(
                    f.virtual_heap.last_leaf_bucket(), 0)
                total_buckets = 0
                for b in range(f.virtual_heap.first_bucket_at_level(0),
                               f.virtual_heap.last_leaf_bucket()+1):
                    orig = f.read_path(b)
                    bucket_path = f.virtual_heap.Node(b).\
                                  bucket_path_from_root()
                    total_buckets += len(bucket_path)
                    self.assertNotEqual(len(bucket_path), 0)
                    self.assertEqual(f.virtual_heap.Node(b).level+1,
                                     len(bucket_path))
                    self.assertEqual(len(orig), len(bucket_path))
                    for i, bucket in zip(bucket_path, orig):
                        self.assertEqual(list(bytearray(bucket)),
                                         list(self._buckets[i]))
                    f.write_path(b, [bytes(data[i])
                                     for i in bucket_path])

                    new = f.read_path(b)
                    self.assertEqual(len(new), len(bucket_path))
                    for i, bucket in zip(bucket_path, new):
                        self.assertEqual(list(bytearray(bucket)),
                                         list(data[i]))

                    f.write_path(b, [bytes(self._buckets[i])
                                     for i in bucket_path])

                    orig = f.read_path(b)
                    self.assertEqual(len(orig), len(bucket_path))
                    for i, bucket in zip(bucket_path, orig):
                        self.assertEqual(list(bytearray(bucket)),
                                         list(self._buckets[i]))

                self.assertEqual(f.bytes_sent,
                                 total_buckets*f.bucket_storage.block_size*2)
                self.assertEqual(f.bytes_received,
                                 total_buckets*f.bucket_storage.block_size*3)
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)

if __name__ == "__main__":
    unittest.main()                                    # pragma: no cover
