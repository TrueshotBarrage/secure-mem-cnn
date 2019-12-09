__all__ = ('EncryptedBlockStorage',)

import struct
import hmac
import hashlib

from pyoram.storage.block_storage import (BlockStorageInterface,
                                          BlockStorageTypeFactory)
from pyoram.crypto.aes import AES

import six

class EncryptedBlockStorageInterface(BlockStorageInterface):

    #
    # Abstract Interface
    #

    @property
    def key(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover
    @property
    def raw_storage(self, *args, **kwds):
        raise NotImplementedError                      # pragma: no cover

class EncryptedBlockStorage(EncryptedBlockStorageInterface):

    _index_struct_string = "!"+("x"*hashlib.sha384().digest_size)+"?"
    _index_offset = struct.calcsize(_index_struct_string)
    _verify_struct_string = "!LLL"
    _verify_size = struct.calcsize(_verify_struct_string)

    def __init__(self, storage, **kwds):
        self._key = kwds.pop('key', None)
        if self._key is None:
            raise ValueError(
                "An encryption key is required using "
                "the 'key' keyword.")
        if isinstance(storage, BlockStorageInterface):
            storage_owned = False
            self._storage = storage
            if len(kwds):
                raise ValueError(
                    "Keywords not used when initializing "
                    "with a storage device: %s"
                    % (str(kwds)))
        else:
            storage_owned = True
            storage_type = kwds.pop('storage_type', 'file')
            self._storage = \
                BlockStorageTypeFactory(storage_type)(storage, **kwds)

        try:
            header_data = AES.GCMDec(self._key,
                                     self._storage.header_data)
            (self._ismodegcm,) = struct.unpack(
                self._index_struct_string,
                header_data[:self._index_offset])
            self._verify_digest = header_data[:hashlib.sha384().digest_size]

            verify = hmac.HMAC(
                key=self.key,
                msg=struct.pack(self._verify_struct_string,
                                self._storage.block_size,
                                self._storage.block_count,
                                len(self._storage.header_data)),
                digestmod=hashlib.sha384)
            if verify.digest() != self._verify_digest:
                raise ValueError(
                    "HMAC of plaintext index data does not match")
            if self._ismodegcm:
                self._encrypt_block_func = AES.GCMEnc
                self._decrypt_block_func = AES.GCMDec
            else:
                self._encrypt_block_func = AES.CTREnc
                self._decrypt_block_func = AES.CTRDec
        except:
            if storage_owned:
                self._storage.close()
            raise

    #
    # Define EncryptedBlockStorageInterface Methods
    #

    @property
    def key(self):
        return self._key

    @property
    def raw_storage(self):
        return self._storage

    #
    # Define BlockStorageInterface Methods
    #

    def clone_device(self):
        return EncryptedBlockStorage(self._storage.clone_device(),
                                     key=self.key)

    @classmethod
    def compute_storage_size(cls,
                             block_size,
                             block_count,
                             aes_mode='ctr',
                             storage_type='file',
                             ignore_header=False,
                             **kwds):
        assert (block_size > 0) and (block_size == int(block_size))
        assert (block_count > 0) and (block_count == int(block_count))
        assert aes_mode in ('ctr', 'gcm')
        if not isinstance(storage_type, BlockStorageInterface):
            storage_type = BlockStorageTypeFactory(storage_type)

        if aes_mode == 'ctr':
            extra_block_data = AES.block_size
        else:
            assert aes_mode == 'gcm'
            extra_block_data = 2 * AES.block_size
        if ignore_header:
            return (extra_block_data * block_count) + \
                    storage_type.compute_storage_size(
                        block_size,
                        block_count,
                        ignore_header=True,
                        **kwds)
        else:
            return cls._index_offset + \
                   2 * AES.block_size + \
                   (extra_block_data * block_count) + \
                   storage_type.compute_storage_size(
                       block_size,
                       block_count,
                       ignore_header=False,
                       **kwds)

    @classmethod
    def setup(cls,
              storage_name,
              block_size,
              block_count,
              aes_mode='ctr',
              key_size=None,
              key=None,
              storage_type='file',
              initialize=None,
              **kwds):

        if (key is not None) and (key_size is not None):
            raise ValueError(
                "Only one of 'key' or 'keysize' keywords can "
                "be specified at a time")
        if key is None:
            if key_size is None:
                key_size = 32
            if key_size not in AES.key_sizes:
                raise ValueError(
                    "Invalid key size: %s" % (key_size))
            key = AES.KeyGen(key_size)
        else:
            if len(key) not in AES.key_sizes:
                raise ValueError(
                    "Invalid key size: %s" % (len(key)))

        if (block_size <= 0) or (block_size != int(block_size)):
            raise ValueError(
                "Block size (bytes) must be a positive integer: %s"
                % (block_size))

        ismodegcm = None
        encrypt_block_func = None
        encrypted_block_size = block_size
        if aes_mode == 'ctr':
            ismodegcm = False
            encrypt_block_func = AES.CTREnc
            encrypted_block_size += AES.block_size
        elif aes_mode == 'gcm':
            ismodegcm = True
            encrypt_block_func = AES.GCMEnc
            encrypted_block_size += (2 * AES.block_size)
        else:
            raise ValueError(
                "AES encryption mode must be one of 'ctr' or 'gcm'. "
                "Invalid value: %s" % (aes_mode))
        assert ismodegcm is not None
        assert encrypt_block_func is not None

        if not isinstance(storage_type, BlockStorageInterface):
            storage_type = BlockStorageTypeFactory(storage_type)

        if initialize is None:
            zeros = bytes(bytearray(block_size))
            initialize = lambda i: zeros
        def encrypted_initialize(i):
            return encrypt_block_func(key, initialize(i))
        kwds['initialize'] = encrypted_initialize

        user_header_data = kwds.get('header_data', bytes())
        if type(user_header_data) is not bytes:
            raise TypeError(
                "'header_data' must be of type bytes. "
                "Invalid type: %s" % (type(user_header_data)))
        # we generate the first time simply to
        # compute the length
        tmp = hmac.HMAC(
            key=key,
            msg=struct.pack(cls._verify_struct_string,
                            encrypted_block_size,
                            block_count,
                            0),
            digestmod=hashlib.sha384).digest()
        header_data = bytearray(struct.pack(cls._index_struct_string,
                                            ismodegcm))
        header_data[:hashlib.sha384().digest_size] = tmp
        header_data = header_data + user_header_data
        header_data = AES.GCMEnc(key, bytes(header_data))
        # now that we know the length of the header data
        # being sent to the underlying storage we can
        # compute the real hmac
        verify_digest = hmac.HMAC(
            key=key,
            msg=struct.pack(cls._verify_struct_string,
                            encrypted_block_size,
                            block_count,
                            len(header_data)),
            digestmod=hashlib.sha384).digest()
        header_data = bytearray(struct.pack(cls._index_struct_string,
                                            ismodegcm))
        header_data[:hashlib.sha384().digest_size] = verify_digest
        header_data = header_data + user_header_data
        kwds['header_data'] = AES.GCMEnc(key, bytes(header_data))

        return EncryptedBlockStorage(
            storage_type.setup(storage_name,
                               encrypted_block_size,
                               block_count,
                               **kwds),
            key=key)

    @property
    def header_data(self):
        return AES.GCMDec(self._key,
                          self._storage.header_data)\
                          [self._index_offset:]

    @property
    def block_count(self):
        return self._storage.block_count

    @property
    def block_size(self):
        if self._ismodegcm:
            return self._storage.block_size - 2 * AES.block_size
        else:
            return self._storage.block_size - AES.block_size

    @property
    def storage_name(self):
        return self._storage.storage_name

    def update_header_data(self, new_header_data):
        self._storage.update_header_data(
            AES.GCMEnc(
                self.key,
                AES.GCMDec(self._key,
                           self._storage.header_data)\
                           [:self._index_offset] + \
                           new_header_data))

    def close(self):
        self._storage.close()

    def read_block(self, i):
        return self._decrypt_block_func(
            self._key,
            self._storage.read_block(i))

    def read_blocks(self, indices, *args, **kwds):
        return [self._decrypt_block_func(self._key, b)
                for b in self._storage.read_blocks(indices, *args, **kwds)]

    def yield_blocks(self, indices, *args, **kwds):
        for b in self._storage.yield_blocks(indices, *args, **kwds):
            yield self._decrypt_block_func(self._key, b)

    def write_block(self, i, block, *args, **kwds):
        self._storage.write_block(
            i,
            self._encrypt_block_func(self._key, block),
            *args, **kwds)

    def write_blocks(self, indices, blocks, *args, **kwds):
        enc_blocks = []
        for i, b in zip(indices, blocks):
            enc_blocks.append(
                self._encrypt_block_func(self._key, b))
        self._storage.write_blocks(indices, enc_blocks, *args, **kwds)

    @property
    def bytes_sent(self):
        return self._storage.bytes_sent

    @property
    def bytes_received(self):
        return self._storage.bytes_received
