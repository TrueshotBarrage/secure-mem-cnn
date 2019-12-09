import os
import unittest
import tempfile
import struct

from pyoram.storage.block_storage import \
    BlockStorageTypeFactory
from pyoram.encrypted_storage.encrypted_block_storage import \
    EncryptedBlockStorage
from pyoram.crypto.aes import AES

from six.moves import xrange

thisdir = os.path.dirname(os.path.abspath(__file__))

class _TestEncryptedBlockStorage(object):

    _type_name = None
    _aes_mode = None
    _test_key = None
    _test_key_size = None

    @classmethod
    def setUpClass(cls):
        assert cls._type_name is not None
        assert cls._aes_mode is not None
        assert not ((cls._test_key is not None) and \
                    (cls._test_key_size is not None))
        fd, cls._dummy_name = tempfile.mkstemp()
        os.close(fd)
        try:
            os.remove(cls._dummy_name)
        except OSError:                                # pragma: no cover
            pass                                       # pragma: no cover
        cls._block_size = 25
        cls._block_count = 5
        cls._testfname = cls.__name__ + "_testfile.bin"
        cls._blocks = []
        f = EncryptedBlockStorage.setup(
            cls._testfname,
            cls._block_size,
            cls._block_count,
            key_size=cls._test_key_size,
            key=cls._test_key,
            storage_type=cls._type_name,
            aes_mode=cls._aes_mode,
            initialize=lambda i: bytes(bytearray([i])*cls._block_size),
            ignore_existing=True)
        f.close()
        cls._key = f.key
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
            EncryptedBlockStorage.setup(
                os.path.join(thisdir,
                             "baselines",
                             "exists.empty"),
                block_size=10,
                block_count=10,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(IOError):
            EncryptedBlockStorage.setup(
                os.path.join(thisdir,
                             "baselines",
                             "exists.empty"),
                block_size=10,
                block_count=10,
                key=self._test_key,
                key_size=self._test_key_size,
                storage_type=self._type_name,
                aes_mode=self._aes_mode,
                ignore_existing=False)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            EncryptedBlockStorage.setup(
                self._dummy_name,
                block_size=0,
                block_count=1,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            EncryptedBlockStorage.setup(
                self._dummy_name,
                block_size=1,
                block_count=0,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(TypeError):
            EncryptedBlockStorage.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=self._aes_mode,
                storage_type=self._type_name,
                header_data=2)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            EncryptedBlockStorage.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                key=self._test_key,
                key_size=self._test_key_size,
                aes_mode=None,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            EncryptedBlockStorage.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                key_size=-1,
                aes_mode=self._aes_mode,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(TypeError):
            EncryptedBlockStorage.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                key=-1,
                aes_mode=self._aes_mode,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            EncryptedBlockStorage.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                key=AES.KeyGen(AES.key_sizes[0]),
                key_size=AES.key_sizes[0],
                aes_mode=self._aes_mode,
                storage_type=self._type_name)
        self.assertEqual(os.path.exists(self._dummy_name), False)
        with self.assertRaises(ValueError):
            EncryptedBlockStorage.setup(
                self._dummy_name,
                block_size=1,
                block_count=1,
                key=os.urandom(AES.key_sizes[0]+100),
                aes_mode=self._aes_mode,
                storage_type=self._type_name)

    def test_setup(self):
        fname = ".".join(self.id().split(".")[1:])
        fname += ".bin"
        fname = os.path.join(thisdir, fname)
        if os.path.exists(fname):
            os.remove(fname)                           # pragma: no cover
        bsize = 10
        bcount = 11
        fsetup = EncryptedBlockStorage.setup(
            fname,
            bsize,
            bcount,
            key=self._test_key,
            key_size=self._test_key_size,
            aes_mode=self._aes_mode,
            storage_type=self._type_name)
        fsetup.close()
        self.assertEqual(type(fsetup.raw_storage),
                         BlockStorageTypeFactory(self._type_name))
        with open(fname, 'rb') as f:
            flen = len(f.read())
            self.assertEqual(
                flen,
                EncryptedBlockStorage.compute_storage_size(
                    bsize,
                    bcount,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name))
            self.assertEqual(
                flen >
                EncryptedBlockStorage.compute_storage_size(
                    bsize,
                    bcount,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name,
                    ignore_header=True),
                True)
        with EncryptedBlockStorage(fname,
                                   key=fsetup.key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.header_data, bytes())
            self.assertEqual(fsetup.header_data, bytes())
            self.assertEqual(f.key, fsetup.key)
            self.assertEqual(f.block_size, bsize)
            self.assertEqual(fsetup.block_size, bsize)
            self.assertEqual(f.block_count, bcount)
            self.assertEqual(fsetup.block_count, bcount)
            self.assertEqual(f.storage_name, fname)
            self.assertEqual(fsetup.storage_name, fname)
        # tamper with the plaintext index
        with open(fname, 'r+b') as f:
            f.seek(0)
            f.write(struct.pack("!L",0))
        with self.assertRaises(ValueError):
            with EncryptedBlockStorage(fname,
                                       key=fsetup.key,
                                       storage_type=self._type_name) as f:
                pass                                       # pragma: no cover
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
        fsetup = EncryptedBlockStorage.setup(
            fname,
            block_size=bsize,
            block_count=bcount,
            key=self._test_key,
            key_size=self._test_key_size,
            aes_mode=self._aes_mode,
            storage_type=self._type_name,
            header_data=header_data)
        fsetup.close()
        self.assertEqual(type(fsetup.raw_storage),
                         BlockStorageTypeFactory(self._type_name))
        with open(fname, 'rb') as f:
            flen = len(f.read())
            self.assertEqual(
                flen,
                EncryptedBlockStorage.compute_storage_size(
                    bsize,
                    bcount,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name,
                    header_data=header_data))
            self.assertTrue(len(header_data) > 0)
            self.assertEqual(
                EncryptedBlockStorage.compute_storage_size(
                    bsize,
                    bcount,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name) <
                EncryptedBlockStorage.compute_storage_size(
                    bsize,
                    bcount,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name,
                    header_data=header_data),
                True)
            self.assertEqual(
                flen >
                EncryptedBlockStorage.compute_storage_size(
                    bsize,
                    bcount,
                    aes_mode=self._aes_mode,
                    storage_type=self._type_name,
                    header_data=header_data,
                    ignore_header=True),
                True)
        with EncryptedBlockStorage(fname,
                                   key=fsetup.key,
                                   storage_type=self._type_name) as f:
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
            with EncryptedBlockStorage(
                    self._dummy_name,
                    key=self._key,
                    storage_type=self._type_name) as f:
                pass                                   # pragma: no cover

    def test_init_exists(self):
        self.assertEqual(os.path.exists(self._testfname), True)
        with open(self._testfname, 'rb') as f:
            databefore = f.read()
        with self.assertRaises(ValueError):
            with EncryptedBlockStorage(self._testfname,
                                       storage_type=self._type_name) as f:
                pass                                   # pragma: no cover
        with self.assertRaises(ValueError):
            with BlockStorageTypeFactory(self._type_name)(self._testfname) as fb:
                with EncryptedBlockStorage(fb,
                                           key=self._key,
                                           storage_type=self._type_name) as f:
                    pass                               # pragma: no cover
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.key, self._key)
            self.assertEqual(f.block_size, self._block_size)
            self.assertEqual(f.block_count, self._block_count)
            self.assertEqual(f.storage_name, self._testfname)
            self.assertEqual(f.header_data, bytes())
        self.assertEqual(os.path.exists(self._testfname), True)
        with open(self._testfname, 'rb') as f:
            dataafter = f.read()
        self.assertEqual(databefore, dataafter)

    def test_read_block(self):
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

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

            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received,
                             self._block_count*f._storage.block_size*4)

        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

            self.assertEqual(list(bytearray(f.read_block(0))),
                             list(self._blocks[0]))
            self.assertEqual(list(bytearray(f.read_block(self._block_count-1))),
                             list(self._blocks[-1]))

            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received,
                             f._storage.block_size*2)

    def test_write_block(self):
        data = bytearray([self._block_count])*self._block_size
        self.assertEqual(len(data) > 0, True)
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

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

            self.assertEqual(f.bytes_sent,
                             self._block_count*f._storage.block_size*2)
            self.assertEqual(f.bytes_received,
                             self._block_count*f._storage.block_size*2)

    def test_read_blocks(self):
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

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

            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received,
                             (2*self._block_count+1)*f._storage.block_size)

    def test_yield_blocks(self):
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

            data = list(f.yield_blocks(list(xrange(self._block_count))))
            self.assertEqual(len(data), self._block_count)
            for i, block in enumerate(data):
                self.assertEqual(list(bytearray(block)),
                                 list(self._blocks[i]))
            data = list(f.yield_blocks([0]))
            self.assertEqual(len(data), 1)
            self.assertEqual(list(bytearray(data[0])),
                             list(self._blocks[0]))
            self.assertEqual(len(self._blocks) > 1, True)
            data = list(f.yield_blocks(list(xrange(1, self._block_count)) + [0]))
            self.assertEqual(len(data), self._block_count)
            for i, block in enumerate(data[:-1], 1):
                self.assertEqual(list(bytearray(block)),
                                 list(self._blocks[i]))
            self.assertEqual(list(bytearray(data[-1])),
                             list(self._blocks[0]))

            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received,
                             (2*self._block_count+1)*f._storage.block_size)

    def test_write_blocks(self):
        data = [bytearray([self._block_count])*self._block_size
                for i in xrange(self._block_count)]
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.bytes_sent, 0)
            self.assertEqual(f.bytes_received, 0)

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

            self.assertEqual(f.bytes_sent,
                             self._block_count*f._storage.block_size*2)
            self.assertEqual(f.bytes_received,
                             self._block_count*f._storage.block_size*3)

    def test_update_header_data(self):
        fname = ".".join(self.id().split(".")[1:])
        fname += ".bin"
        fname = os.path.join(thisdir, fname)
        if os.path.exists(fname):
            os.remove(fname)                           # pragma: no cover
        bsize = 10
        bcount = 11
        header_data = bytes(bytearray([0,1,2]))
        fsetup = EncryptedBlockStorage.setup(
            fname,
            block_size=bsize,
            block_count=bcount,
            key=self._test_key,
            key_size=self._test_key_size,
            header_data=header_data)
        fsetup.close()
        new_header_data = bytes(bytearray([1,1,1]))
        with EncryptedBlockStorage(fname,
                                   key=fsetup.key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.header_data, header_data)
            f.update_header_data(new_header_data)
            self.assertEqual(f.header_data, new_header_data)
        with EncryptedBlockStorage(fname,
                                   key=fsetup.key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.header_data, new_header_data)
        with self.assertRaises(ValueError):
            with EncryptedBlockStorage(fname,
                                       key=fsetup.key,
                                       storage_type=self._type_name) as f:
                f.update_header_data(bytes(bytearray([1,1])))
        with self.assertRaises(ValueError):
            with EncryptedBlockStorage(fname,
                                       key=fsetup.key,
                                       storage_type=self._type_name) as f:
                f.update_header_data(bytes(bytearray([1,1,1,1])))
        with EncryptedBlockStorage(fname,
                                   key=fsetup.key,
                                   storage_type=self._type_name) as f:
            self.assertEqual(f.header_data, new_header_data)
        os.remove(fname)

    def test_locked_flag(self):
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as f:
            with self.assertRaises(IOError):
                with EncryptedBlockStorage(self._testfname,
                                           key=self._key,
                                           storage_type=self._type_name) as f1:
                    pass                               # pragma: no cover
            with self.assertRaises(IOError):
                with EncryptedBlockStorage(self._testfname,
                                           key=self._key,
                                           storage_type=self._type_name) as f1:
                    pass                               # pragma: no cover
            with EncryptedBlockStorage(self._testfname,
                                       key=self._key,
                                       storage_type=self._type_name,
                                       ignore_lock=True) as f1:
                pass
            with self.assertRaises(IOError):
                with EncryptedBlockStorage(self._testfname,
                                           key=self._key,
                                           storage_type=self._type_name) as f1:
                    pass                               # pragma: no cover
            with EncryptedBlockStorage(self._testfname,
                                       key=self._key,
                                       storage_type=self._type_name,
                                       ignore_lock=True) as f1:
                pass
            with EncryptedBlockStorage(self._testfname,
                                       key=self._key,
                                       storage_type=self._type_name,
                                       ignore_lock=True) as f1:
                pass
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as f:
            pass

    def test_read_block_cloned(self):
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as forig:
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)
            with forig.clone_device() as f:
                self.assertEqual(forig.bytes_sent, 0)
                self.assertEqual(forig.bytes_received, 0)
                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received, 0)

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

                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received,
                                 self._block_count*f._storage.block_size*4)

            with forig.clone_device() as f:
                self.assertEqual(forig.bytes_sent, 0)
                self.assertEqual(forig.bytes_received, 0)
                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received, 0)

                self.assertEqual(list(bytearray(f.read_block(0))),
                                 list(self._blocks[0]))
                self.assertEqual(list(bytearray(f.read_block(self._block_count-1))),
                                 list(self._blocks[-1]))

                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received,
                                 f._storage.block_size*2)

            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)

    def test_write_block_cloned(self):
        data = bytearray([self._block_count])*self._block_size
        self.assertEqual(len(data) > 0, True)
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as forig:
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)
            with forig.clone_device() as f:
                self.assertEqual(forig.bytes_sent, 0)
                self.assertEqual(forig.bytes_received, 0)
                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received, 0)

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

                self.assertEqual(f.bytes_sent,
                                 self._block_count*f._storage.block_size*2)
                self.assertEqual(f.bytes_received,
                                 self._block_count*f._storage.block_size*2)
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)

    def test_read_blocks_cloned(self):
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as forig:
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)
            with forig.clone_device() as f:
                self.assertEqual(forig.bytes_sent, 0)
                self.assertEqual(forig.bytes_received, 0)
                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received, 0)

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

                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received,
                                 (2*self._block_count + 1)*f._storage.block_size)
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)

    def test_yield_blocks_cloned(self):
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as forig:
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)
            with forig.clone_device() as f:
                self.assertEqual(forig.bytes_sent, 0)
                self.assertEqual(forig.bytes_received, 0)
                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received, 0)

                data = list(f.yield_blocks(list(xrange(self._block_count))))
                self.assertEqual(len(data), self._block_count)
                for i, block in enumerate(data):
                    self.assertEqual(list(bytearray(block)),
                                     list(self._blocks[i]))
                data = list(f.yield_blocks([0]))
                self.assertEqual(len(data), 1)
                self.assertEqual(list(bytearray(data[0])),
                                 list(self._blocks[0]))
                self.assertEqual(len(self._blocks) > 1, True)
                data = list(f.yield_blocks(list(xrange(1, self._block_count)) + [0]))
                self.assertEqual(len(data), self._block_count)
                for i, block in enumerate(data[:-1], 1):
                    self.assertEqual(list(bytearray(block)),
                                     list(self._blocks[i]))
                self.assertEqual(list(bytearray(data[-1])),
                                 list(self._blocks[0]))

                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received,
                                 (2*self._block_count + 1)*f._storage.block_size)
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)

    def test_write_blocks_cloned(self):
        data = [bytearray([self._block_count])*self._block_size
                for i in xrange(self._block_count)]
        with EncryptedBlockStorage(self._testfname,
                                   key=self._key,
                                   storage_type=self._type_name) as forig:
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)
            with forig.clone_device() as f:
                self.assertEqual(forig.bytes_sent, 0)
                self.assertEqual(forig.bytes_received, 0)
                self.assertEqual(f.bytes_sent, 0)
                self.assertEqual(f.bytes_received, 0)

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

                self.assertEqual(f.bytes_sent,
                                 self._block_count*f._storage.block_size*2)
                self.assertEqual(f.bytes_received,
                                 self._block_count*f._storage.block_size*3)
            self.assertEqual(forig.bytes_sent, 0)
            self.assertEqual(forig.bytes_received, 0)

class TestEncryptedBlockStorageFileCTRKey(_TestEncryptedBlockStorage,
                                          unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'ctr'
    _test_key = AES.KeyGen(16)

class TestEncryptedBlockStorageFileCTR32(_TestEncryptedBlockStorage,
                                         unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'ctr'
    _test_key_size = 16

class TestEncryptedBlockStorageFileGCMKey(_TestEncryptedBlockStorage,
                                          unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'gcm'
    _test_key = AES.KeyGen(24)

class TestEncryptedBlockStorageFileGCM32(_TestEncryptedBlockStorage,
                                         unittest.TestCase):
    _type_name = 'file'
    _aes_mode = 'gcm'
    _test_key_size = 24

class TestEncryptedBlockStorageMMapFileCTRKey(_TestEncryptedBlockStorage,
                                              unittest.TestCase):
    _type_name = 'mmap'
    _aes_mode = 'ctr'
    _test_key = AES.KeyGen(32)

class TestEncryptedBlockStorageMMapFileCTR32(_TestEncryptedBlockStorage,
                                             unittest.TestCase):
    _type_name = 'mmap'
    _aes_mode = 'ctr'
    _test_key_size = 32

class TestEncryptedBlockStorageMMapFileGCMKey(_TestEncryptedBlockStorage,
                                              unittest.TestCase):
    _type_name = 'mmap'
    _aes_mode = 'gcm'
    _test_key = AES.KeyGen(32)

class TestEncryptedBlockStorageMMapFileGCM32(_TestEncryptedBlockStorage,
                                             unittest.TestCase):
    _type_name = 'mmap'
    _aes_mode = 'gcm'
    _test_key_size = 32

if __name__ == "__main__":
    unittest.main()                                    # pragma: no cover
