__all__ = ("VirtualHeap",
           "SizedVirtualHeap")

import os
import sys
import subprocess
import random
import string
import tempfile

from six.moves import xrange

from pyoram.util._virtual_heap_helper import lib as _clib
from pyoram.util.misc import log2floor

numerals = ''.join([c for c in string.printable \
                  if ((c not in string.whitespace) and \
                      (c != '+') and (c != '-') and \
                      (c != '"') and (c != "'") and \
                      (c != '\\') and (c != '/'))])
numeral_index = dict((c,i) for i,c in enumerate(numerals))

# The maximum heap base for which base k labels
# can be produced.
max_k_labeled = len(numerals)

def base10_integer_to_basek_string(k, x):
    """Convert an integer into a base k string."""
    if not (2 <= k <= max_k_labeled):
        raise ValueError("k must be in range [2, %d]: %s"
                         % (max_k_labeled, k))
    return ((x == 0) and numerals[0]) or \
        (base10_integer_to_basek_string(k, x // k).\
         lstrip(numerals[0]) + numerals[x % k])

def basek_string_to_base10_integer(k, x):
    """Convert a base k string into an integer."""
    assert 1 < k <= max_k_labeled
    return sum(numeral_index[c]*(k**i)
               for i, c in enumerate(reversed(x)))

# _clib defines a faster version of this function
def calculate_bucket_level(k, b):
    """
    Calculate the level in which a 0-based bucket
    lives inside of a k-ary heap.
    """
    assert k >= 2
    if k == 2:
        return log2floor(b+1)
    v = (k - 1) * (b + 1) + 1
    h = 0
    while k**(h+1) < v:
        h += 1
    return h

# _clib defines a faster version of this function
def calculate_last_common_level(k, b1, b2):
    """
    Calculate the highest level after which the
    paths from the root to these buckets diverge.
    """
    l1 = calculate_bucket_level(k, b1)
    l2 = calculate_bucket_level(k, b2)
    while l1 > l2:
        b1 = (b1-1)//k
        l1 -= 1
    while l2 > l1:
        b2 = (b2-1)//k
        l2 -= 1
    while b1 != b2:
        b1 = (b1-1)//k
        b2 = (b2-1)//k
        l1 -= 1
    return l1

def calculate_necessary_heap_height(k, n):
    """
    Calculate the necessary k-ary heap height
    to store n buckets.
    """
    assert n >= 1
    return calculate_bucket_level(k, n-1)

def calculate_bucket_count_in_heap_with_height(k, h):
    """
    Calculate the number of buckets in a
    k-ary heap of height h.
    """
    assert h >= 0
    return ((k**(h+1)) - 1) // (k - 1)

def calculate_bucket_count_in_heap_at_level(k, l):
    """
    Calculate the number of buckets in a
    k-ary heap at level l.
    """
    assert l >= 0
    return k**l

def calculate_leaf_bucket_count_in_heap_with_height(k, h):
    """
    Calculate the number of buckets in the
    leaf-level of a k-ary heap of height h.
    """
    return calculate_bucket_count_in_heap_at_level(k, h)

def create_node_type(k):

    class VirtualHeapNode(object):
        __slots__ = ("bucket", "level")
        def __init__(self, bucket):
            assert bucket >= 0
            self.bucket = bucket
            self.level = _clib.calculate_bucket_level(self.k, self.bucket)

        def __hash__(self):
            return self.bucket.__hash__()
        def __int__(self):
            return self.bucket
        def __lt__(self, other):
            return self.bucket < other
        def __le__(self, other):
            return self.bucket <= other
        def __eq__(self, other):
            return self.bucket == other
        def __ne__(self, other):
            return self.bucket != other
        def __gt__(self, other):
            return self.bucket > other
        def __ge__(self, other):
            return self.bucket >= other
        def last_common_level(self, n):
            return _clib.calculate_last_common_level(self.k,
                                                     self.bucket,
                                                     n.bucket)
        def child_node(self, c):
            assert type(c) is int
            assert 0 <= c < self.k
            return VirtualHeapNode(self.k * self.bucket + 1 + c)
        def parent_node(self):
            if self.bucket != 0:
                return VirtualHeapNode((self.bucket - 1)//self.k)
            return None
        def ancestor_node_at_level(self, level):
            if level > self.level:
                return None
            current = self
            while current.level != level:
                current = current.parent_node()
            return current
        def path_to_root(self):
            bucket = self.bucket
            yield self
            while bucket != 0:
                bucket = (bucket - 1)//self.k
                yield type(self)(bucket)
        def path_from_root(self):
            return list(reversed(list(self.path_to_root())))
        def bucket_path_to_root(self):
            bucket = self.bucket
            yield bucket
            while bucket != 0:
                bucket = (bucket - 1)//self.k
                yield bucket
        def bucket_path_from_root(self):
            return list(reversed(list(self.bucket_path_to_root())))

        #
        # Expensive Functions
        #
        def __repr__(self):
            try:
                label = self.label()
            except ValueError:
                # presumably, k is too large
                label = "<unknown>"
            return ("VirtualHeapNode(k=%s, bucket=%s, level=%s, label=%r)"
                    % (self.k, self.bucket, self.level, label))
        def __str__(self):
            """Returns a tuple (<level>, <bucket offset within level>)."""
            if self.bucket != 0:
                return ("(%s, %s)"
                        % (self.level,
                           self.bucket -
                           calculate_bucket_count_in_heap_with_height(self.k,
                                                                self.level-1)))
            assert self.level == 0
            return "(0, 0)"

        def label(self):
            assert 0 <= self.bucket
            if self.level == 0:
                return ''
            b_offset = self.bucket - \
                       calculate_bucket_count_in_heap_with_height(self.k,
                                                            self.level-1)
            basek = base10_integer_to_basek_string(self.k, b_offset)
            return basek.zfill(self.level)

        def is_node_on_path(self, n):
            if n.level <= self.level:
                n_label = n.label()
                if n_label == "":
                    return True
                return self.label().startswith(n_label)
            return False

    VirtualHeapNode.k = k

    return VirtualHeapNode

class VirtualHeap(object):

    clib = _clib
    random = random.SystemRandom()

    def __init__(self, k, blocks_per_bucket=1):
        assert 1 < k
        assert blocks_per_bucket >= 1
        self._k = k
        self._blocks_per_bucket = blocks_per_bucket
        self.Node = create_node_type(k)

    @property
    def k(self):
        return self._k

    def node_label_to_bucket(self, label):
        if len(label) > 0:
            return \
                (calculate_bucket_count_in_heap_with_height(self.k,
                                                      len(label)-1) +
                 basek_string_to_base10_integer(self.k, label))
        return 0

    #
    # Buckets (0-based integer, equivalent to block for heap
    # with blocks_per_bucket=1)
    #

    @property
    def blocks_per_bucket(self):
        return self._blocks_per_bucket

    def bucket_count_at_level(self, l):
        return calculate_bucket_count_in_heap_at_level(self.k, l)
    def first_bucket_at_level(self, l):
        if l > 0:
            return calculate_bucket_count_in_heap_with_height(self.k, l-1)
        return 0
    def last_bucket_at_level(self, l):
        return calculate_bucket_count_in_heap_with_height(self.k, l) - 1
    def random_bucket_up_to_level(self, l):
        return self.random.randint(self.first_bucket_at_level(0),
                                   self.last_bucket_at_level(l))
    def random_bucket_at_level(self, l):
        return self.random.randint(self.first_bucket_at_level(l),
                                   self.first_bucket_at_level(l+1)-1)

    #
    # Nodes (a class that helps with heap path calculations)
    #

    def root_node(self):
        return self.first_node_at_level(0)
    def node_count_at_level(self, l):
        return self.bucket_count_at_level(l)
    def first_node_at_level(self, l):
        return self.Node(self.first_bucket_at_level(l))
    def last_node_at_level(self, l):
        return self.Node(self.last_bucket_at_level(l))
    def random_node_up_to_level(self, l):
        return self.Node(self.random_bucket_up_to_level(l))
    def random_node_at_level(self, l):
        return self.Node(self.random_bucket_at_level(l))

    #
    # Block (0-based integer)
    #

    def bucket_to_block(self, b):
        assert b >= 0
        return b * self.blocks_per_bucket
    def block_to_bucket(self, s):
        assert s >= 0
        return s//self.blocks_per_bucket
    def first_block_in_bucket(self, b):
        return self.bucket_to_block(b)
    def last_block_in_bucket(self, b):
        return self.bucket_to_block(b) + self.blocks_per_bucket - 1
    def block_count_at_level(self, l):
        return self.bucket_count_at_level(l) * self.blocks_per_bucket
    def first_block_at_level(self, l):
        return self.bucket_to_block(self.first_bucket_at_level(l))
    def last_block_at_level(self, l):
        return self.bucket_to_block(self.first_bucket_at_level(l+1)) - 1

class SizedVirtualHeap(VirtualHeap):

    def __init__(self, k, height, blocks_per_bucket=1):
        super(SizedVirtualHeap, self).\
            __init__(k, blocks_per_bucket=blocks_per_bucket)
        self._height = height

    #
    # Size properties
    #
    @property
    def height(self):
        return self._height
    @property
    def levels(self):
        return self.height + 1
    @property
    def first_level(self):
        return 0
    @property
    def last_level(self):
        return self.height

    #
    # Buckets (0-based integer, equivalent to block for heap
    # with blocks_per_bucket=1)
    #

    def bucket_count(self):
        return calculate_bucket_count_in_heap_with_height(self.k,
                                                          self.height)
    def leaf_bucket_count(self):
        return calculate_leaf_bucket_count_in_heap_with_height(self.k,
                                                               self.height)
    def first_leaf_bucket(self):
        return self.first_bucket_at_level(self.height)
    def last_leaf_bucket(self):
        return self.last_bucket_at_level(self.height)
    def random_bucket(self):
        return self.random.randint(self.first_bucket_at_level(0),
                                   self.last_leaf_bucket())
    def random_leaf_bucket(self):
        return self.random_bucket_at_level(self.height)

    #
    # Nodes (a class that helps with heap path calculations)
    #

    def is_nil_node(self, n):
        return n.bucket >= self.bucket_count()
    def node_count(self):
        return self.bucket_count()
    def leaf_node_count(self):
        return self.leaf_bucket_count()
    def first_leaf_node(self):
        return self.Node(self.first_leaf_bucket())
    def last_leaf_node(self):
        return self.Node(self.last_leaf_bucket())
    def random_leaf_node(self):
        return self.Node(self.random_leaf_bucket())
    def random_node(self):
        return self.Node(self.random_bucket())

    #
    # Block (0-based integer)
    #

    def block_count(self):
        return self.bucket_count() * self.blocks_per_bucket
    def leaf_block_count(self):
        return self.leaf_bucket_count() * self.blocks_per_bucket
    def first_leaf_block(self):
        return self.first_block_in_bucket(self.first_leaf_bucket())
    def last_leaf_block(self):
        return self.last_block_in_bucket(self.last_leaf_bucket())

    #
    # Visualization
    #

    def write_as_dot(self, f, data=None, max_levels=None):
        "Write the tree in the dot language format to f."
        assert (max_levels is None) or (max_levels >= 0)
        def visit_node(n, levels):
            lbl = "{"
            if data is None:
                if self.k <= max_k_labeled:
                    lbl = repr(n.label()).\
                          replace("{","\{").\
                          replace("}","\}").\
                          replace("|","\|").\
                          replace("<","\<").\
                          replace(">","\>")
                else:
                    lbl = str(n)
            else:
                s = self.bucket_to_block(n.bucket)
                for i in xrange(self.blocks_per_bucket):
                    lbl += "{%s}" % (data[s+i])
                    if i + 1 != self.blocks_per_bucket:
                        lbl += "|"
            lbl += "}"
            f.write("  %s [penwidth=%s,label=\"%s\"];\n"
                    % (n.bucket, 1, lbl))
            levels += 1
            if (max_levels is None) or (levels <= max_levels):
                for i in xrange(self.k):
                    cn = n.child_node(i)
                    if not self.is_nil_node(cn):
                        visit_node(cn, levels)
                        f.write("  %s -> %s ;\n" % (n.bucket, cn.bucket))

        f.write("// Created by SizedVirtualHeap.write_as_dot(...)\n")
        f.write("digraph heaptree {\n")
        f.write("node [shape=record]\n")

        if (max_levels is None) or (max_levels > 0):
            visit_node(self.root_node(), 1)
        f.write("}\n")

    def save_image_as_pdf(self, filename, data=None, max_levels=None):
        "Write the heap as PDF file."
        assert (max_levels is None) or (max_levels >= 0)
        import os
        if not filename.endswith('.pdf'):
            filename = filename+'.pdf'
        tmpfd, tmpname = tempfile.mkstemp(suffix='dot')
        with open(tmpname, 'w') as f:
            self.write_as_dot(f, data=data, max_levels=max_levels)
        os.close(tmpfd)
        try:
            subprocess.call(['dot',
                             tmpname,
                             '-Tpdf',
                             '-o',
                             ('%s'%filename)])
        except OSError:
            sys.stderr.write(
                "DOT -> PDF conversion failed. See DOT file: %s\n"
                % (tmpname))
            return False
        os.remove(tmpname)
        return True
