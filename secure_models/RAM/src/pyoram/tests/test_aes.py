import unittest

from pyoram.crypto.aes import AES

class TestAES(unittest.TestCase):

    def test_KeyGen(self):
        self.assertTrue(len(AES.key_sizes) in (3,4))
        self.assertTrue(len(set(AES.key_sizes)) in (3,4))
        for keysize in AES.key_sizes:
            key_list = []
            key_set = set()
            for i in range(10):
                k = AES.KeyGen(keysize)
                self.assertEqual(len(k), keysize)
                key_list.append(k)
                key_set.add(k)
            self.assertEqual(len(key_list), 10)
            # make sure every key is unique
            self.assertEqual(len(key_list), len(key_set))

    def test_CTR(self):
        self._test_Enc_Dec(
            AES.CTREnc,
            AES.CTRDec,
            lambda i, size: bytes(bytearray([i]) * size),
            [16,24,32])

    def test_GCM(self):
        self._test_Enc_Dec(
            AES.GCMEnc,
            AES.GCMDec,
            lambda i, size: bytes(bytearray([i]) * size),
            [16,24,32])

    def _test_Enc_Dec(self,
                      enc_func,
                      dec_func,
                      get_plaintext,
                      keysizes):
        keysizes = list(keysizes)
        self.assertTrue(len(keysizes) > 0)
        blocksize_factor = [0.5, 1, 1.5, 2, 2.5]
        plaintext_blocks = []
        for i, f in enumerate(blocksize_factor):
            size = AES.block_size * f
            size = int(round(size))
            if int(f) != f:
                assert (size % AES.block_size) != 0
            plaintext_blocks.append(get_plaintext(i, size))

        assert len(AES.key_sizes) > 0
        ciphertext_blocks = {}
        keys = {}
        for keysize in keysizes:
            key = AES.KeyGen(keysize)
            keys[keysize] = key
            ciphertext_blocks[keysize] = []
            for block in plaintext_blocks:
                ciphertext_blocks[keysize].append(
                    enc_func(key, block))

        self.assertEqual(len(ciphertext_blocks),
                         len(keysizes))
        self.assertEqual(len(keys),
                         len(keysizes))

        plaintext_decrypted_blocks = {}
        for keysize in keys:
            key = keys[keysize]
            plaintext_decrypted_blocks[keysize] = []
            for block in ciphertext_blocks[keysize]:
                plaintext_decrypted_blocks[keysize].append(
                    dec_func(key, block))

        self.assertEqual(len(plaintext_decrypted_blocks),
                         len(keysizes))

        for i in range(len(blocksize_factor)):
            for keysize in keysizes:
                self.assertEqual(
                    plaintext_blocks[i],
                    plaintext_decrypted_blocks[keysize][i])
                self.assertNotEqual(
                    plaintext_blocks[i],
                    ciphertext_blocks[keysize][i])
                if enc_func is AES.CTREnc:
                    self.assertEqual(
                        len(ciphertext_blocks[keysize][i]),
                        len(plaintext_blocks[i]) + AES.block_size)
                else:
                    assert enc_func is AES.GCMEnc
                    self.assertEqual(
                        len(ciphertext_blocks[keysize][i]),
                        len(plaintext_blocks[i]) + 2*AES.block_size)
                # check IND-CPA
                key = keys[keysize]
                alt_ciphertext = enc_func(key, plaintext_blocks[i])
                self.assertNotEqual(
                    ciphertext_blocks[keysize][i],
                    alt_ciphertext)
                self.assertEqual(
                    len(ciphertext_blocks[keysize][i]),
                    len(alt_ciphertext))
                self.assertNotEqual(
                    ciphertext_blocks[keysize][i][:AES.block_size],
                    alt_ciphertext[:AES.block_size])
                self.assertNotEqual(
                    ciphertext_blocks[keysize][i][AES.block_size:],
                    alt_ciphertext[AES.block_size:])

if __name__ == "__main__":
    unittest.main()                                    # pragma: no cover
