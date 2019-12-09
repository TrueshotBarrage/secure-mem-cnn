import base64

import six

def log2floor(n):
    """
    Returns the exact value of floor(log2(n)).
    No floating point calculations are used.
    Requires positive integer type.
    """
    assert n > 0
    return n.bit_length() - 1

def log2ceil(n):
    """
    Returns the exact value of ceil(log2(n)).
    No floating point calculations are used.
    Requires positive integer type.
    """
    if n == 1:
        return 0
    return log2floor(n-1) + 1

def intdivceil(x, y):
    """
    Returns the exact value of ceil(x // y).
    No floating point calculations are used.
    Requires positive integer types. The result
    is undefined if at least one of the inputs
    is floating point.
    """
    result = x // y
    if (x % y):
        result += 1
    return result

def save_private_key(filename, key):
    with open(filename, "wb") as f:
        f.write(base64.b64encode(key))

def load_private_key(filename):
    with open(filename, "rb") as f:
        return base64.b64decode(f.read())

from fractions import Fraction

class MemorySize(object):

    to_bytes = {}
    to_bytes['b'] = lambda x: Fraction(x,8)
    to_bytes['B'] = lambda x: Fraction(x,1)
    to_bytes['KB'] = lambda x: Fraction(1000*x,1)
    to_bytes['MB'] = lambda x: Fraction((1000**2)*x,1)
    to_bytes['GB'] = lambda x: Fraction((1000**3)*x,1)
    to_bytes['TB'] = lambda x: Fraction((1000**4)*x,1)
    to_bytes['KiB'] = lambda x: Fraction(1024*x,1)
    to_bytes['MiB'] = lambda x: Fraction((1024**2)*x,1)
    to_bytes['GiB'] = lambda x: Fraction((1024**3)*x,1)
    to_bytes['TiB'] = lambda x: Fraction((1024**4)*x,1)

    def __init__(self, size, unit='B'):
        assert size >= 0
        self.numbytes = MemorySize.to_bytes[unit](Fraction.from_float(size))

    def __str__(self):
        if self.B < 1:
            return "%.3f b" % (self.b)
        if self.KB < 1:
            return "%.3f B" % (self.B)
        if self.MB < 1:
            return "%.3f KB" % (self.KB)
        if self.GB < 1:
            return "%.3f MB" % (self.MB)
        if self.TB < 1:
            return "%.3f GB" % (self.GB)
        return "%.3f TB" % (self.TB)

    @property
    def b(self): return self.numbytes*8
    @property
    def B(self): return self.numbytes

    @property
    def KB(self): return self.B/1000
    @property
    def MB(self): return self.KB/1000
    @property
    def GB(self): return self.MB/1000
    @property
    def TB(self): return self.GB/1000

    @property
    def KiB(self): return self.B/1024
    @property
    def MiB(self): return self.KiB/1024
    @property
    def GiB(self): return self.MiB/1024
    @property
    def TiB(self): return self.GiB/1024

def chunkiter(objs, n=100):
    """
    Chunk an iterator of unknown size. The optional
    keyword 'n' sets the chunk size (default 100).
    """

    objs = iter(objs)
    try:
        while (1):
            chunk = []
            while len(chunk) < n:
                chunk.append(six.next(objs))
            yield chunk
    except StopIteration:
        pass
    if len(chunk):
        yield chunk
