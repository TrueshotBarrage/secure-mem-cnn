import os
import unittest
import tempfile

import pyoram.util.misc

class Test(unittest.TestCase):

    def test_log2floor(self):
        self.assertEqual(pyoram.util.misc.log2floor(1), 0)
        self.assertEqual(pyoram.util.misc.log2floor(2), 1)
        self.assertEqual(pyoram.util.misc.log2floor(3), 1)
        self.assertEqual(pyoram.util.misc.log2floor(4), 2)
        self.assertEqual(pyoram.util.misc.log2floor(5), 2)
        self.assertEqual(pyoram.util.misc.log2floor(6), 2)
        self.assertEqual(pyoram.util.misc.log2floor(7), 2)
        self.assertEqual(pyoram.util.misc.log2floor(8), 3)
        self.assertEqual(pyoram.util.misc.log2floor(9), 3)

    def test_log2ceil(self):
        self.assertEqual(pyoram.util.misc.log2ceil(1), 0)
        self.assertEqual(pyoram.util.misc.log2ceil(2), 1)
        self.assertEqual(pyoram.util.misc.log2ceil(3), 2)
        self.assertEqual(pyoram.util.misc.log2ceil(4), 2)
        self.assertEqual(pyoram.util.misc.log2ceil(5), 3)
        self.assertEqual(pyoram.util.misc.log2ceil(6), 3)
        self.assertEqual(pyoram.util.misc.log2ceil(7), 3)
        self.assertEqual(pyoram.util.misc.log2ceil(8), 3)
        self.assertEqual(pyoram.util.misc.log2ceil(9), 4)

    def test_intdivceil(self):

        with self.assertRaises(ZeroDivisionError):
            pyoram.util.misc.intdivceil(0, 0)
        with self.assertRaises(ZeroDivisionError):
            pyoram.util.misc.intdivceil(1, 0)

        self.assertEqual(pyoram.util.misc.intdivceil(1, 1), 1)
        self.assertEqual(pyoram.util.misc.intdivceil(2, 3), 1)
        self.assertEqual(2 // 3, 0)
        self.assertEqual(pyoram.util.misc.intdivceil(
            123123123123123123123123123123123123123123123123,
            123123123123123123123123123123123123123123123123), 1)
        self.assertEqual(pyoram.util.misc.intdivceil(
            2 * 123123123123123123123123123123123123123123123123,
            123123123123123123123123123123123123123123123123), 2)
        self.assertEqual(pyoram.util.misc.intdivceil(
            2 * 123123123123123123123123123123123123123123123123 + 1,
            123123123123123123123123123123123123123123123123), 3)
        self.assertEqual(pyoram.util.misc.intdivceil(
            2 * 123123123123123123123123123123123123123123123123 - 1,
            123123123123123123123123123123123123123123123123), 2)
        self.assertEqual(
            (2 * 123123123123123123123123123123123123123123123123 - 1) // \
            123123123123123123123123123123123123123123123123,
            1)

    def test_MemorySize(self):
        self.assertTrue("b" in str(pyoram.util.misc.MemorySize(0.1)))
        self.assertTrue("B" in str(pyoram.util.misc.MemorySize(1)))
        self.assertTrue("B" in str(pyoram.util.misc.MemorySize(999)))
        self.assertTrue("KB" in str(pyoram.util.misc.MemorySize(1000)))
        self.assertTrue("KB" in str(pyoram.util.misc.MemorySize(999999)))
        self.assertTrue("MB" in str(pyoram.util.misc.MemorySize(1000000)))
        self.assertTrue("MB" in str(pyoram.util.misc.MemorySize(999999999)))
        self.assertTrue("GB" in str(pyoram.util.misc.MemorySize(1000000000)))
        self.assertTrue("GB" in str(pyoram.util.misc.MemorySize(9999999999)))
        self.assertTrue("TB" in str(pyoram.util.misc.MemorySize(1000000000000)))
        self.assertTrue("b" in str(pyoram.util.misc.MemorySize(1, unit="b")))
        self.assertTrue("b" in str(pyoram.util.misc.MemorySize(2, unit="b")))
        self.assertTrue("b" in str(pyoram.util.misc.MemorySize(7.9, unit="b")))

        self.assertTrue("B" in str(pyoram.util.misc.MemorySize(8, unit="b")))
        self.assertTrue("B" in str(pyoram.util.misc.MemorySize(1, unit="B")))
        self.assertTrue("B" in str(pyoram.util.misc.MemorySize(999, unit="B")))

        self.assertTrue("KB" in str(pyoram.util.misc.MemorySize(1000, unit="B")))
        self.assertTrue("KB" in str(pyoram.util.misc.MemorySize(1, unit="KB")))
        self.assertTrue("KB" in str(pyoram.util.misc.MemorySize(999, unit="KB")))
        self.assertTrue("MB" in str(pyoram.util.misc.MemorySize(1000, unit="KB")))
        self.assertTrue("MB" in str(pyoram.util.misc.MemorySize(1, unit="MB")))
        self.assertTrue("MB" in str(pyoram.util.misc.MemorySize(999, unit="MB")))
        self.assertTrue("GB" in str(pyoram.util.misc.MemorySize(1000, unit="MB")))
        self.assertTrue("GB" in str(pyoram.util.misc.MemorySize(1, unit="GB")))
        self.assertTrue("GB" in str(pyoram.util.misc.MemorySize(999, unit="GB")))
        self.assertTrue("TB" in str(pyoram.util.misc.MemorySize(1000, unit="GB")))
        self.assertTrue("TB" in str(pyoram.util.misc.MemorySize(1, unit="TB")))

        self.assertEqual(pyoram.util.misc.MemorySize(1024).KiB, 1)
        self.assertEqual(pyoram.util.misc.MemorySize(1024**2).MiB, 1)
        self.assertEqual(pyoram.util.misc.MemorySize(1024**3).GiB, 1)
        self.assertEqual(pyoram.util.misc.MemorySize(1024**4).TiB, 1)

    def test_saveload_private_key(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        try:
            key = os.urandom(32)
            pyoram.util.misc.save_private_key(filename, key)
            loaded_key = pyoram.util.misc.load_private_key(filename)
            self.assertEqual(key, loaded_key)
        finally:
            os.remove(filename)

    def test_chunkiter(self):
        self.assertEqual(list(pyoram.util.misc.chunkiter([1,2,3,4,5], 1)),
                         [[1],[2],[3],[4],[5]])
        self.assertEqual(list(pyoram.util.misc.chunkiter([1,2,3,4,5], 2)),
                         [[1,2],[3,4],[5]])
        self.assertEqual(list(pyoram.util.misc.chunkiter([1,2,3,4,5], 3)),
                         [[1,2,3],[4,5]])
        self.assertEqual(list(pyoram.util.misc.chunkiter([1,2,3,4,5], 4)),
                         [[1,2,3,4],[5]])
        self.assertEqual(list(pyoram.util.misc.chunkiter([1,2,3,4,5], 5)),
                         [[1,2,3,4,5]])
        self.assertEqual(list(pyoram.util.misc.chunkiter([1,2,3,4,5], 6)),
                         [[1,2,3,4,5]])
        self.assertEqual(list(pyoram.util.misc.chunkiter([], 1)),
                         [])
        self.assertEqual(list(pyoram.util.misc.chunkiter([], 2)),
                         [])

if __name__ == "__main__":
    unittest.main()                                    # pragma: no cover
