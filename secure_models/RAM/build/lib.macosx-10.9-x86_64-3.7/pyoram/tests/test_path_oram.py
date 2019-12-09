import os
import unittest
import tempfile

from pyoram.oblivious_storage.tree.path_oram import \
    PathORAM
from pyoram.storage.block_storage import \
    BlockStorageTypeFactory
from pyoram.encrypted_storage.encrypted_heap_storage import \
    EncryptedHeapStorage
from pyoram.crypto.aes import AES

from six.moves import xrange

thisdir = os.path.dirname(os.path.abspath(__file__))

class _TestPathORAMBase(object):

    _type_name = None
    _aes_mode = None
    _test_key = None
    _test_key_size = None
    _bucket_capacity = None
    _heap_base = None
    _kwds = None

    @classmethod
    def setUpClass(cls):
        assert cls._type_name is not None
        assert cls._aes_mode is not None
        assert not ((cls._test_key is not None) and \
                    (cls._test_key_size is not None))
        assert cls._bucket_capacity is not None
        assert cls._heap_base is not None
        assert cls._kwds is not None
        fd, cls._dummy_name = tempfile.mkstemp()
        os.close(fd)
        try:
            os.remove(cls._dummy_name)
        except OSError:                                # pragma: no cover
            pass                                       # pragma: no cover
        cls._block_size = 25
        cls._block_count = 47
        cls._testfname = cls.__name__ + "_testfile.bin"
        cls._blocks = []
        f = PathORAM.setup(
            cls._testfname,
            cls._block_size,
            cls._block_count,
            bucket_capacity=cls._bucket_capacity,
            heap_base=cls._heap_base,
            key_size=cls._test_key_size,
            key=cls._test_key,
            storage_type=cls._type_name,
            aes_mode=cls._aes_mode,
            initialize=lambda i: bytes(bytearray([i])*cls._block_size),
            ignore_existing=True,
            **cls._kwds)
        f.close()
        cls._key = f.key
        cls._stash = f.stash
        cls._position_map = f.position_map
        for i in range(cls._block_count):
            data = bytearray([i])*cls._block_size
            cls._blocks.append(data)

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
            PathORAM.setup(
                os.path.join(thisdir,
                             "baselines",
                             "exists.empty"),
                block_size=10,
                block_count=10,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(IOError):
            PathORAM.setup(
                os.path.join(thisdir,
                             "baselines",
                             "exists.empty"),
                block_size=10,
                block_count=10,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=self._test_key,
                key_size=self._test_key_size,
                storage_type=self._type_name,
                aes_mode=self._aes_mode,
                ignore_existing=False,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=0,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            PathORAM.setup(
                self._dummy_name,
                block_size=0,
                block_count=1,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(TypeError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                header_data=2,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=None,
                storage_type=self._type_name,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                bucket_capacity=0,
                heap_base=self._heap_base,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                bucket_capacity=self._bucket_capacity,
                heap_base=1,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key_size=-1,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(TypeError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=-1,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=AES.KeyGen(AES.key_sizes[-1]),
                key_size=AES.key_sizes[-1],
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=os.urandom(AES.key_sizes[-1]+100),
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)
        with self.assertRaises(ValueError):
            PathORAM.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                heap_height=1,
                bucket_capacity=self._bucket_capacity,
                heap_base=self._heap_base,
                key=self._key,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                **self._kwds)

    def test_setup(self):
        fname = ".".join(self.id().split(".")[1:])
        fname += ".bin"
        fname = os.path.join(thisdir, fname)
        if os.path.exists(fname):
            os.remove(fname)                           # pragma: no cover
        bsize = 10
        bcount = 11
        fsetup = PathORAM.setup(
            fname,
            bsize,
            bcount,
            bucket_capacity=self._bucket_capacity,
            heap_base=self._heap_base,
            key=self._test_key,
            key_size=self._test_key_size,
            aes_mode=self._aes_mode,
            storage_type=self._type_name,
            **self._kwds)
        fsetup.close()
        self.assertEqual(type(fsetup.raw_storage),
                         BlockStorageTypeFactory(self._type_name))
        # test that these can be called with default keyword values
        fsetup.stash_digest(fsetup.stash)
        fsetup.position_map_digest(fsetup.position_map)
        with open(fname, 'rb') as f:
            flen = len(f.read())
            self.assertEqual(
                flen,
                PathORAM.compute_storage_size(
                    bsize,
                    bcount,
                    bucket_capacity=self._bucket_capacity,
                    heap_base=self._heap_base,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name))
            self.assertEqual(
                flen >
                PathORAM.compute_storage_size(
                    bsize,
                    bcount,
                    bucket_capacity=self._bucket_capacity,
                    heap_base=self._heap_base,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name,
                    ignore_header=True),
                True)
        with PathORAM(fname,
                      fsetup.stash,
                      fsetup.position_map,
                      key=fsetup.key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            self.assertEqual(f.header_data, bytes())
            self.assertEqual(fsetup.header_data, bytes())
            self.assertEqual(f.key, fsetup.key)
            self.assertEqual(f.block_size, bsize)
            self.assertEqual(fsetup.block_size, bsize)
            self.assertEqual(f.block_count, bcount)
            self.assertEqual(fsetup.block_count, bcount)
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
        bcount = 11
        header_data = bytes(bytearray([0,1,2]))
        fsetup = PathORAM.setup(
            fname,
            block_size=bsize,
            block_count=bcount,
            bucket_capacity=self._bucket_capacity,
            heap_base=self._heap_base,
            key=self._test_key,
            key_size=self._test_key_size,
            aes_mode=self._aes_mode,
            storage_type=self._type_name,
            header_data=header_data,
            **self._kwds)
        fsetup.close()
        self.assertEqual(type(fsetup.raw_storage),
                         BlockStorageTypeFactory(self._type_name))
        with open(fname, 'rb') as f:
            flen = len(f.read())
            self.assertEqual(
                flen,
                PathORAM.compute_storage_size(
                    bsize,
                    bcount,
                    bucket_capacity=self._bucket_capacity,
                    heap_base=self._heap_base,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name,
                    header_data=header_data))
            self.assertTrue(len(header_data) > 0)
            self.assertEqual(
                PathORAM.compute_storage_size(
                    bsize,
                    bcount,
                    bucket_capacity=self._bucket_capacity,
                    heap_base=self._heap_base,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name) <
                PathORAM.compute_storage_size(
                    bsize,
                    bcount,
                    bucket_capacity=self._bucket_capacity,
                    heap_base=self._heap_base,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name,
                    header_data=header_data),
                True)
            self.assertEqual(
                flen >
                PathORAM.compute_storage_size(
                    bsize,
                    bcount,
                    bucket_capacity=self._bucket_capacity,
                    heap_base=self._heap_base,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name,
                    header_data=header_data,
                    ignore_header=True),
                True)
        with PathORAM(fname,
                      fsetup.stash,
                      fsetup.position_map,
                      key=fsetup.key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            self.assertEqual(f.header_data, header_data)
            self.assertEqual(fsetup.header_data, header_data)
            self.assertEqual(f.key, fsetup.key)
            self.assertEqual(f.block_size, bsize)
            self.assertEqual(fsetup.block_size, bsize)
            self.assertEqual(f.block_count, bcount)
            self.assertEqual(fsetup.block_count, bcount)
            self.assertEqual(f.storage_name, fname)
            self.assertEqual(fsetup.storage_name, fname)
        os.remove(fname)

    def test_init_noexists(self):
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(IOError):
            with PathORAM(
                    self._dummy_name,
                    self._stash,
                    self._position_map,
                    key=self._key,
                    storage_type=self._type_name,
                    **self._kwds) as f:
                pass                                   # pragma: no cover

    def test_init_exists(self):
        self.assertEqual(os.path.exists(self._testfname), True)
        with open(self._testfname, 'rb') as f:
            databefore = f.read()
        # no key
        with self.assertRaises(ValueError):
            with PathORAM(self._testfname,
                          self._stash,
                          self._position_map,
                          storage_type=self._type_name,
                          **self._kwds) as f:
                pass                                   # pragma: no cover
        # stash does not match digest
        with self.assertRaises(ValueError):
            with PathORAM(self._testfname,
                          {1: bytes()},
                          self._position_map,
                          key=self._key,
                          storage_type=self._type_name,
                          **self._kwds) as f:
                pass                                   # pragma: no cover
        # stash hash invalid key (negative)
        with self.assertRaises(ValueError):
            with PathORAM(self._testfname,
                          {-1: bytes()},
                          self._position_map,
                          key=self._key,
                          storage_type=self._type_name,
                          **self._kwds) as f:
                pass                                   # pragma: no cover
        # position map has invalid item (negative)
        with self.assertRaises(ValueError):
            with PathORAM(self._testfname,
                          self._stash,
                          [-1],
                          key=self._key,
                          storage_type=self._type_name,
                          **self._kwds) as f:
                pass                                   # pragma: no cover
        # position map does not match digest
        with self.assertRaises(ValueError):
            with PathORAM(self._testfname,
                          self._stash,
                          [1],
                          key=self._key,
                          storage_type=self._type_name,
                          **self._kwds) as f:
                pass                                   # pragma: no cover
        with self.assertRaises(ValueError):
            with EncryptedHeapStorage(self._testfname,
                                      key=self._key,
                                      storage_type=self._type_name) as fb:
                with PathORAM(fb,
                              self._stash,
                              self._position_map,
                              key=self._key,
                              storage_type=self._type_name,
                              **self._kwds) as f:
                    self.assertIs(f.heap_storage, fb)
                    pass                               # pragma: no cover
        with PathORAM(self._testfname,
                      self._stash,
                      self._position_map,
                      key=self._key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            self.assertEqual(f.key, self._key)
            self.assertEqual(f.block_size, self._block_size)
            self.assertEqual(f.block_count, self._block_count)
            self.assertEqual(f.storage_name, self._testfname)
            self.assertEqual(f.header_data, bytes())
        self.assertEqual(os.path.exists(self._testfname), True)
        with open(self._testfname, 'rb') as f:
            dataafter = f.read()
        self.assertEqual(databefore[-(self._block_count*self._block_size):],
                         dataafter[-(self._block_count*self._block_size):])

    def test_read_block(self):
        with PathORAM(self._testfname,
                      self._stash,
                      self._position_map,
                      key=self._key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            for i, data in enumerate(self._blocks):
                self.assertEqual(list(bytearray(f.read_block(i))),
                                 list(self._blocks[i]))
            for i, data in enumerate(self._blocks):
                self.assertEqual(list(bytearray(f.read_block(i))),
                                 list(self._blocks[i]))
            for i, data in reversed(list(enumerate(self._blocks))):
                self.assertEqual(list(bytearray(f.read_block(i))),
                                 list(self._blocks[i]))
            for i, data in reversed(list(enumerate(self._blocks))):
                self.assertEqual(list(bytearray(f.read_block(i))),
                                 list(self._blocks[i]))
        with PathORAM(self._testfname,
                      self._stash,
                      self._position_map,
                      key=self._key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            self.assertEqual(list(bytearray(f.read_block(0))),
                             list(self._blocks[0]))
            self.assertEqual(list(bytearray(f.read_block(self._block_count-1))),
                             list(self._blocks[-1]))

        # test eviction behavior of the tree oram helper
        with PathORAM(self._testfname,
                      self._stash,
                      self._position_map,
                      key=self._key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            oram = f._oram
            vheap = oram.storage_heap.virtual_heap
            Z = vheap.blocks_per_bucket
            def _has_vacancies(level):
                return any(oram.path_block_ids[i] == oram.empty_block_id
                           for i in range(level*Z, (level+1)*Z))

            for i in range(len(f.position_map)):
                b = f.position_map[i]
                f.position_map[i] = vheap.random_leaf_bucket()
                oram.load_path(b)
                block = oram.extract_block_from_path(i)
                if block is not None:
                    oram.stash[i] = block

                # track where everyone should be able to move
                # to, unless the bucket fills up
                eviction_levels = {}
                for id_, level in zip(oram.path_block_ids,
                                      oram.path_block_eviction_levels):
                    eviction_levels[id_] = level
                for id_ in oram.stash:
                    block_id, block_addr = \
                        oram.get_block_info(oram.stash[id_])
                    assert block_id == id_
                    eviction_levels[id_] = \
                        vheap.clib.calculate_last_common_level(
                            vheap.k, b, block_addr)

                oram.push_down_path()
                oram.fill_path_from_stash()
                oram.evict_path()

                # check that everyone was pushed down greedily
                oram.load_path(b)
                for pos, id_ in enumerate(oram.path_block_ids):
                    current_level = pos // Z
                    if (id_ != oram.empty_block_id):
                        eviction_level = eviction_levels[id_]
                        self.assertEqual(current_level <= eviction_level, True)
                        if current_level < eviction_level:
                            self.assertEqual(_has_vacancies(eviction_level), False)
                for id_ in oram.stash:
                    self.assertEqual(
                        _has_vacancies(eviction_levels[id_]), False)

    def test_write_block(self):
        data = bytearray([self._block_count])*self._block_size
        self.assertEqual(len(data) > 0, True)
        with PathORAM(self._testfname,
                      self._stash,
                      self._position_map,
                      key=self._key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            for i in xrange(self._block_count):
                self.assertNotEqual(list(bytearray(f.read_block(i))),
                                    list(data))
            for i in xrange(self._block_count):
                f.write_block(i, bytes(data))
            for i in xrange(self._block_count):
                self.assertEqual(list(bytearray(f.read_block(i))),
                                 list(data))
            for i, block in enumerate(self._blocks):
                f.write_block(i, bytes(block))

    def test_read_blocks(self):
        with PathORAM(self._testfname,
                      self._stash,
                      self._position_map,
                      key=self._key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            data = f.read_blocks(list(xrange(self._block_count)))
            self.assertEqual(len(data), self._block_count)
            for i, block in enumerate(data):
                self.assertEqual(list(bytearray(block)),
                                 list(self._blocks[i]))
            data = f.read_blocks([0])
            self.assertEqual(len(data), 1)
            self.assertEqual(list(bytearray(data[0])),
                             list(self._blocks[0]))
            self.assertEqual(len(self._blocks) > 1, True)
            data = f.read_blocks(list(xrange(1, self._block_count)) + [0])
            self.assertEqual(len(data), self._block_count)
            for i, block in enumerate(data[:-1], 1):
                self.assertEqual(list(bytearray(block)),
                                 list(self._blocks[i]))
            self.assertEqual(list(bytearray(data[-1])),
                             list(self._blocks[0]))

    def test_write_blocks(self):
        data = [bytearray([self._block_count])*self._block_size
                for i in xrange(self._block_count)]
        with PathORAM(self._testfname,
                      self._stash,
                      self._position_map,
                      key=self._key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            orig = f.read_blocks(list(xrange(self._block_count)))
            self.assertEqual(len(orig), self._block_count)
            for i, block in enumerate(orig):
                self.assertEqual(list(bytearray(block)),
                                 list(self._blocks[i]))
            f.write_blocks(list(xrange(self._block_count)),
                           [bytes(b) for b in data])
            new = f.read_blocks(list(xrange(self._block_count)))
            self.assertEqual(len(new), self._block_count)
            for i, block in enumerate(new):
                self.assertEqual(list(bytearray(block)),
                                 list(data[i]))
            f.write_blocks(list(xrange(self._block_count)),
                           [bytes(b) for b in self._blocks])
            orig = f.read_blocks(list(xrange(self._block_count)))
            self.assertEqual(len(orig), self._block_count)
            for i, block in enumerate(orig):
                self.assertEqual(list(bytearray(block)),
                                 list(self._blocks[i]))

    def test_update_header_data(self):
        fname = ".".join(self.id().split(".")[1:])
        fname += ".bin"
        fname = os.path.join(thisdir, fname)
        if os.path.exists(fname):
            os.remove(fname)                           # pragma: no cover
        bsize = 10
        bcount = 11
        header_data = bytes(bytearray([0,1,2]))
        fsetup = PathORAM.setup(
            fname,
            block_size=bsize,
            block_count=bcount,
            bucket_capacity=self._bucket_capacity,
            heap_base=self._heap_base,
            key=self._test_key,
            key_size=self._test_key_size,
            header_data=header_data,
            **self._kwds)
        fsetup.close()
        new_header_data = bytes(bytearray([1,1,1]))
        with PathORAM(fname,
                      fsetup.stash,
                      fsetup.position_map,
                      key=fsetup.key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            self.assertEqual(f.header_data, header_data)
            f.update_header_data(new_header_data)
            self.assertEqual(f.header_data, new_header_data)
        with PathORAM(fname,
                      fsetup.stash,
                      fsetup.position_map,
                      key=fsetup.key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            self.assertEqual(f.header_data, new_header_data)
        with self.assertRaises(ValueError):
            with PathORAM(fname,
                          fsetup.stash,
                          fsetup.position_map,
                          key=fsetup.key,
                          storage_type=self._type_name,
                          **self._kwds) as f:
                f.update_header_data(bytes(bytearray([1,1])))
        with self.assertRaises(ValueError):
            with PathORAM(fname,
                          fsetup.stash,
                          fsetup.position_map,
                          key=fsetup.key,
                          storage_type=self._type_name,
                          **self._kwds) as f:
                f.update_header_data(bytes(bytearray([1,1,1,1])))
        with PathORAM(fname,
                      fsetup.stash,
                      fsetup.position_map,
                      key=fsetup.key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            self.assertEqual(f.header_data, new_header_data)
        os.remove(fname)

    def test_locked_flag(self):
        with PathORAM(self._testfname,
                      self._stash,
                      self._position_map,
                      key=self._key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            with self.assertRaises(IOError):
                with PathORAM(self._testfname,
                              self._stash,
                              self._position_map,
                              key=self._key,
                              storage_type=self._type_name,
                              **self._kwds) as f1:
                    pass                               # pragma: no cover
            with self.assertRaises(IOError):
                with PathORAM(self._testfname,
                              self._stash,
                              self._position_map,
                              key=self._key,
                              storage_type=self._type_name,
                              **self._kwds) as f1:
                    pass                               # pragma: no cover
            with PathORAM(self._testfname,
                          self._stash,
                          self._position_map,
                          key=self._key,
                          storage_type=self._type_name,
                          ignore_lock=True,
                          **self._kwds) as f1:
                pass
            with self.assertRaises(IOError):
                with PathORAM(self._testfname,
                              self._stash,
                              self._position_map,
                              key=self._key,
                              storage_type=self._type_name,
                              **self._kwds) as f1:
                    pass                               # pragma: no cover
            with PathORAM(self._testfname,
                          self._stash,
                          self._position_map,
                          key=self._key,
                          storage_type=self._type_name,
                          ignore_lock=True,
                          **self._kwds) as f1:
                pass
            with PathORAM(self._testfname,
                          self._stash,
                          self._position_map,
                          key=self._key,
                          storage_type=self._type_name,
                          ignore_lock=True,
                          **self._kwds) as f1:
                pass
        with PathORAM(self._testfname,
                      self._stash,
                      self._position_map,
                      key=self._key,
                      storage_type=self._type_name,
                      **self._kwds) as f:
            pass

class TestPathORAMB2Z1(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'ctr'
    _bucket_capacity = 1
    _heap_base = 2
    _kwds = {'cached_levels': 0}

class TestPathORAMB2Z2(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'mmap'
    _aes_mode = 'gcm'
    _bucket_capacity = 2
    _heap_base = 2
    _kwds = {'cached_levels': 0}

class TestPathORAMB2Z3(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'mmap'
    _aes_mode = 'ctr'
    _bucket_capacity = 3
    _heap_base = 2
    _kwds = {'cached_levels': 1}

class TestPathORAMB2Z4(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'gcm'
    _bucket_capacity = 4
    _heap_base = 2
    _kwds = {'cached_levels': 1}

class TestPathORAMB2Z5(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'ctr'
    _bucket_capacity = 5
    _heap_base = 2
    _kwds = {'cached_levels': 2,
             'concurrency_level': 0}

class TestPathORAMB3Z1(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'ctr'
    _bucket_capacity = 1
    _heap_base = 3
    _kwds = {'cached_levels': 2,
             'concurrency_level': 1}

class TestPathORAMB3Z2(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'mmap'
    _aes_mode = 'gcm'
    _bucket_capacity = 2
    _heap_base = 3
    _kwds = {}

class TestPathORAMB3Z3(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'mmap'
    _aes_mode = 'ctr'
    _bucket_capacity = 3
    _heap_base = 3
    _kwds = {}

class TestPathORAMB3Z4(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'gcm'
    _bucket_capacity = 4
    _heap_base = 3
    _kwds = {}

class TestPathORAMB3Z5(_TestPathORAMBase,
                       unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'ctr'
    _bucket_capacity = 5
    _heap_base = 3
    _kwds = {}

if __name__ == "__main__":
    unittest.main()                                    # pragma: no cover
