import os
import subprocess
import unittest

import pyoram
from pyoram.util.virtual_heap import \
    (VirtualHeap,
     SizedVirtualHeap,
     max_k_labeled,
     calculate_bucket_count_in_heap_with_height,
     calculate_bucket_count_in_heap_at_level,
     calculate_bucket_level,
     calculate_last_common_level,
     calculate_necessary_heap_height,
     basek_string_to_base10_integer,
     numerals,
     _clib)

from six.moves import xrange

thisdir = os.path.dirname(os.path.abspath(__file__))
baselinedir = os.path.join(thisdir, "baselines")

try:
    has_dot = not subprocess.call(['dot','-?'],
                                  stdout=subprocess.PIPE)
except:                                                # pragma: no cover
    has_dot = False

_test_bases = list(xrange(2, 15)) + [max_k_labeled+1]
_test_labeled_bases = list(xrange(2, 15)) + [max_k_labeled]

def _do_preorder(x):
    if x.level > 2:
        return
    yield x.bucket
    for c in xrange(x.k):
        for b in _do_preorder(x.child_node(c)):
            yield b

def _do_postorder(x):
    if x.level > 2:
        return
    for c in xrange(x.k):
        for b in _do_postorder(x.child_node(c)):
            yield b
    yield x.bucket

def _do_inorder(x):
    assert x.k == 2
    if x.level > 2:
        return
    for b in _do_inorder(x.child_node(0)):
        yield b
    yield x.bucket
    for b in _do_inorder(x.child_node(1)):
        yield b

class TestVirtualHeapNode(unittest.TestCase):

    def test_init(self):
        for k in _test_bases:
            vh = VirtualHeap(k)
            node = vh.Node(0)
            self.assertEqual(node.k, k)
            self.assertEqual(node.bucket, 0)
            self.assertEqual(node.level, 0)
            for b in xrange(1, k+1):
                node = vh.Node(b)
                self.assertEqual(node.k, k)
                self.assertEqual(node.bucket, b)
                self.assertEqual(node.level, 1)

    def test_level(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(Node(0).level, 0)
        self.assertEqual(Node(1).level, 1)
        self.assertEqual(Node(2).level, 1)
        self.assertEqual(Node(3).level, 2)
        self.assertEqual(Node(4).level, 2)
        self.assertEqual(Node(5).level, 2)
        self.assertEqual(Node(6).level, 2)
        self.assertEqual(Node(7).level, 3)

        Node = VirtualHeap(3).Node
        self.assertEqual(Node(0).level, 0)
        self.assertEqual(Node(1).level, 1)
        self.assertEqual(Node(2).level, 1)
        self.assertEqual(Node(3).level, 1)
        self.assertEqual(Node(4).level, 2)
        self.assertEqual(Node(5).level, 2)
        self.assertEqual(Node(6).level, 2)
        self.assertEqual(Node(7).level, 2)
        self.assertEqual(Node(8).level, 2)
        self.assertEqual(Node(9).level, 2)
        self.assertEqual(Node(10).level, 2)
        self.assertEqual(Node(11).level, 2)
        self.assertEqual(Node(12).level, 2)
        self.assertEqual(Node(13).level, 3)

    def test_hash(self):
        x1 = VirtualHeap(3).Node(5)
        x2 = VirtualHeap(2).Node(5)
        self.assertNotEqual(id(x1), id(x2))
        self.assertEqual(x1, x2)
        self.assertEqual(x1, x1)
        self.assertEqual(x2, x2)

        all_node_set = set()
        all_node_list = list()
        for k in _test_bases:
            node_set = set()
            node_list = list()
            Node = VirtualHeap(k).Node
            for height in xrange(k+2):
                node = Node(height)
                node_set.add(node)
                all_node_set.add(node)
                node_list.append(node)
                all_node_list.append(node)
            self.assertEqual(sorted(node_set),
                             sorted(node_list))
        self.assertNotEqual(sorted(all_node_set),
                            sorted(all_node_list))
    def test_int(self):
        Node2 = VirtualHeap(2).Node
        Node3 = VirtualHeap(3).Node
        for b in range(100):
            self.assertEqual(int(Node2(b)), b)
            self.assertEqual(int(Node3(b)), b)

    def test_lt(self):
        Node = VirtualHeap(3).Node
        self.assertEqual(Node(5) < 4, False)
        self.assertEqual(Node(5) < 5, False)
        self.assertEqual(Node(5) < 6, True)

    def test_le(self):
        Node = VirtualHeap(3).Node
        self.assertEqual(Node(5) <= 4, False)
        self.assertEqual(Node(5) <= 5, True)
        self.assertEqual(Node(5) <= 6, True)

    def test_eq(self):
        Node = VirtualHeap(3).Node
        self.assertEqual(Node(5) == 4, False)
        self.assertEqual(Node(5) == 5, True)
        self.assertEqual(Node(5) == 6, False)

    def test_ne(self):
        Node = VirtualHeap(3).Node
        self.assertEqual(Node(5) != 4, True)
        self.assertEqual(Node(5) != 5, False)
        self.assertEqual(Node(5) != 6, True)

    def test_gt(self):
        Node = VirtualHeap(3).Node
        self.assertEqual(Node(5) > 4, True)
        self.assertEqual(Node(5) > 5, False)
        self.assertEqual(Node(5) > 6, False)

    def test_ge(self):
        Node = VirtualHeap(3).Node
        self.assertEqual(Node(5) >= 4, True)
        self.assertEqual(Node(5) >= 5, True)
        self.assertEqual(Node(5) >= 6, False)

    def test_last_common_level_k2(self):
        Node = VirtualHeap(2).Node
        n0 = Node(0)
        n1 = Node(1)
        n2 = Node(2)
        n3 = Node(3)
        n4 = Node(4)
        n5 = Node(5)
        n6 = Node(6)
        n7 = Node(7)
        self.assertEqual(n0.last_common_level(n0), 0)
        self.assertEqual(n0.last_common_level(n1), 0)
        self.assertEqual(n0.last_common_level(n2), 0)
        self.assertEqual(n0.last_common_level(n3), 0)
        self.assertEqual(n0.last_common_level(n4), 0)
        self.assertEqual(n0.last_common_level(n5), 0)
        self.assertEqual(n0.last_common_level(n6), 0)
        self.assertEqual(n0.last_common_level(n7), 0)

        self.assertEqual(n1.last_common_level(n0), 0)
        self.assertEqual(n1.last_common_level(n1), 1)
        self.assertEqual(n1.last_common_level(n2), 0)
        self.assertEqual(n1.last_common_level(n3), 1)
        self.assertEqual(n1.last_common_level(n4), 1)
        self.assertEqual(n1.last_common_level(n5), 0)
        self.assertEqual(n1.last_common_level(n6), 0)
        self.assertEqual(n1.last_common_level(n7), 1)

        self.assertEqual(n2.last_common_level(n0), 0)
        self.assertEqual(n2.last_common_level(n1), 0)
        self.assertEqual(n2.last_common_level(n2), 1)
        self.assertEqual(n2.last_common_level(n3), 0)
        self.assertEqual(n2.last_common_level(n4), 0)
        self.assertEqual(n2.last_common_level(n5), 1)
        self.assertEqual(n2.last_common_level(n6), 1)
        self.assertEqual(n2.last_common_level(n7), 0)

        self.assertEqual(n3.last_common_level(n0), 0)
        self.assertEqual(n3.last_common_level(n1), 1)
        self.assertEqual(n3.last_common_level(n2), 0)
        self.assertEqual(n3.last_common_level(n3), 2)
        self.assertEqual(n3.last_common_level(n4), 1)
        self.assertEqual(n3.last_common_level(n5), 0)
        self.assertEqual(n3.last_common_level(n6), 0)
        self.assertEqual(n3.last_common_level(n7), 2)

        self.assertEqual(n4.last_common_level(n0), 0)
        self.assertEqual(n4.last_common_level(n1), 1)
        self.assertEqual(n4.last_common_level(n2), 0)
        self.assertEqual(n4.last_common_level(n3), 1)
        self.assertEqual(n4.last_common_level(n4), 2)
        self.assertEqual(n4.last_common_level(n5), 0)
        self.assertEqual(n4.last_common_level(n6), 0)
        self.assertEqual(n4.last_common_level(n7), 1)

        self.assertEqual(n5.last_common_level(n0), 0)
        self.assertEqual(n5.last_common_level(n1), 0)
        self.assertEqual(n5.last_common_level(n2), 1)
        self.assertEqual(n5.last_common_level(n3), 0)
        self.assertEqual(n5.last_common_level(n4), 0)
        self.assertEqual(n5.last_common_level(n5), 2)
        self.assertEqual(n5.last_common_level(n6), 1)
        self.assertEqual(n5.last_common_level(n7), 0)

        self.assertEqual(n6.last_common_level(n0), 0)
        self.assertEqual(n6.last_common_level(n1), 0)
        self.assertEqual(n6.last_common_level(n2), 1)
        self.assertEqual(n6.last_common_level(n3), 0)
        self.assertEqual(n6.last_common_level(n4), 0)
        self.assertEqual(n6.last_common_level(n5), 1)
        self.assertEqual(n6.last_common_level(n6), 2)
        self.assertEqual(n6.last_common_level(n7), 0)

        self.assertEqual(n7.last_common_level(n0), 0)
        self.assertEqual(n7.last_common_level(n1), 1)
        self.assertEqual(n7.last_common_level(n2), 0)
        self.assertEqual(n7.last_common_level(n3), 2)
        self.assertEqual(n7.last_common_level(n4), 1)
        self.assertEqual(n7.last_common_level(n5), 0)
        self.assertEqual(n7.last_common_level(n6), 0)
        self.assertEqual(n7.last_common_level(n7), 3)

    def test_last_common_level_k3(self):
        Node = VirtualHeap(3).Node
        n0 = Node(0)
        n1 = Node(1)
        n2 = Node(2)
        n3 = Node(3)
        n4 = Node(4)
        n5 = Node(5)
        n6 = Node(6)
        n7 = Node(7)
        self.assertEqual(n0.last_common_level(n0), 0)
        self.assertEqual(n0.last_common_level(n1), 0)
        self.assertEqual(n0.last_common_level(n2), 0)
        self.assertEqual(n0.last_common_level(n3), 0)
        self.assertEqual(n0.last_common_level(n4), 0)
        self.assertEqual(n0.last_common_level(n5), 0)
        self.assertEqual(n0.last_common_level(n6), 0)
        self.assertEqual(n0.last_common_level(n7), 0)

        self.assertEqual(n1.last_common_level(n0), 0)
        self.assertEqual(n1.last_common_level(n1), 1)
        self.assertEqual(n1.last_common_level(n2), 0)
        self.assertEqual(n1.last_common_level(n3), 0)
        self.assertEqual(n1.last_common_level(n4), 1)
        self.assertEqual(n1.last_common_level(n5), 1)
        self.assertEqual(n1.last_common_level(n6), 1)
        self.assertEqual(n1.last_common_level(n7), 0)

        self.assertEqual(n2.last_common_level(n0), 0)
        self.assertEqual(n2.last_common_level(n1), 0)
        self.assertEqual(n2.last_common_level(n2), 1)
        self.assertEqual(n2.last_common_level(n3), 0)
        self.assertEqual(n2.last_common_level(n4), 0)
        self.assertEqual(n2.last_common_level(n5), 0)
        self.assertEqual(n2.last_common_level(n6), 0)
        self.assertEqual(n2.last_common_level(n7), 1)

        self.assertEqual(n3.last_common_level(n0), 0)
        self.assertEqual(n3.last_common_level(n1), 0)
        self.assertEqual(n3.last_common_level(n2), 0)
        self.assertEqual(n3.last_common_level(n3), 1)
        self.assertEqual(n3.last_common_level(n4), 0)
        self.assertEqual(n3.last_common_level(n5), 0)
        self.assertEqual(n3.last_common_level(n6), 0)
        self.assertEqual(n3.last_common_level(n7), 0)

        self.assertEqual(n4.last_common_level(n0), 0)
        self.assertEqual(n4.last_common_level(n1), 1)
        self.assertEqual(n4.last_common_level(n2), 0)
        self.assertEqual(n4.last_common_level(n3), 0)
        self.assertEqual(n4.last_common_level(n4), 2)
        self.assertEqual(n4.last_common_level(n5), 1)
        self.assertEqual(n4.last_common_level(n6), 1)
        self.assertEqual(n4.last_common_level(n7), 0)

        self.assertEqual(n5.last_common_level(n0), 0)
        self.assertEqual(n5.last_common_level(n1), 1)
        self.assertEqual(n5.last_common_level(n2), 0)
        self.assertEqual(n5.last_common_level(n3), 0)
        self.assertEqual(n5.last_common_level(n4), 1)
        self.assertEqual(n5.last_common_level(n5), 2)
        self.assertEqual(n5.last_common_level(n6), 1)
        self.assertEqual(n5.last_common_level(n7), 0)

        self.assertEqual(n6.last_common_level(n0), 0)
        self.assertEqual(n6.last_common_level(n1), 1)
        self.assertEqual(n6.last_common_level(n2), 0)
        self.assertEqual(n6.last_common_level(n3), 0)
        self.assertEqual(n6.last_common_level(n4), 1)
        self.assertEqual(n6.last_common_level(n5), 1)
        self.assertEqual(n6.last_common_level(n6), 2)
        self.assertEqual(n6.last_common_level(n7), 0)

        self.assertEqual(n7.last_common_level(n0), 0)
        self.assertEqual(n7.last_common_level(n1), 0)
        self.assertEqual(n7.last_common_level(n2), 1)
        self.assertEqual(n7.last_common_level(n3), 0)
        self.assertEqual(n7.last_common_level(n4), 0)
        self.assertEqual(n7.last_common_level(n5), 0)
        self.assertEqual(n7.last_common_level(n6), 0)
        self.assertEqual(n7.last_common_level(n7), 2)

    def test_child_node(self):
        root = VirtualHeap(2).Node(0)
        self.assertEqual(list(_do_preorder(root)),
                         [0, 1, 3, 4, 2, 5, 6])
        self.assertEqual(list(_do_postorder(root)),
                         [3, 4, 1, 5, 6, 2, 0])
        self.assertEqual(list(_do_inorder(root)),
                         [3, 1, 4, 0, 5, 2, 6])

        root = VirtualHeap(3).Node(0)
        self.assertEqual(
            list(_do_preorder(root)),
            [0, 1, 4, 5, 6, 2, 7, 8, 9, 3, 10, 11, 12])
        self.assertEqual(
            list(_do_postorder(root)),
            [4, 5, 6, 1, 7, 8, 9, 2, 10, 11, 12, 3, 0])

    def test_parent_node(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(Node(0).parent_node(),
                         None)
        self.assertEqual(Node(1).parent_node(),
                         Node(0))
        self.assertEqual(Node(2).parent_node(),
                         Node(0))
        self.assertEqual(Node(3).parent_node(),
                         Node(1))
        self.assertEqual(Node(4).parent_node(),
                         Node(1))
        self.assertEqual(Node(5).parent_node(),
                         Node(2))
        self.assertEqual(Node(6).parent_node(),
                         Node(2))
        self.assertEqual(Node(7).parent_node(),
                         Node(3))

        Node = VirtualHeap(3).Node
        self.assertEqual(Node(0).parent_node(),
                         None)
        self.assertEqual(Node(1).parent_node(),
                         Node(0))
        self.assertEqual(Node(2).parent_node(),
                         Node(0))
        self.assertEqual(Node(3).parent_node(),
                         Node(0))
        self.assertEqual(Node(4).parent_node(),
                         Node(1))
        self.assertEqual(Node(5).parent_node(),
                         Node(1))
        self.assertEqual(Node(6).parent_node(),
                         Node(1))
        self.assertEqual(Node(7).parent_node(),
                         Node(2))
        self.assertEqual(Node(8).parent_node(),
                         Node(2))
        self.assertEqual(Node(9).parent_node(),
                         Node(2))
        self.assertEqual(Node(10).parent_node(),
                         Node(3))
        self.assertEqual(Node(11).parent_node(),
                         Node(3))
        self.assertEqual(Node(12).parent_node(),
                         Node(3))
        self.assertEqual(Node(13).parent_node(),
                         Node(4))

    def test_ancestor_node_at_level(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(Node(0).ancestor_node_at_level(0),
                         Node(0))
        self.assertEqual(Node(0).ancestor_node_at_level(1),
                         None)
        self.assertEqual(Node(1).ancestor_node_at_level(0),
                         Node(0))
        self.assertEqual(Node(1).ancestor_node_at_level(1),
                         Node(1))
        self.assertEqual(Node(1).ancestor_node_at_level(2),
                         None)
        self.assertEqual(Node(3).ancestor_node_at_level(0),
                         Node(0))
        self.assertEqual(Node(3).ancestor_node_at_level(1),
                         Node(1))
        self.assertEqual(Node(3).ancestor_node_at_level(2),
                         Node(3))
        self.assertEqual(Node(3).ancestor_node_at_level(3),
                         None)

        Node = VirtualHeap(3).Node
        self.assertEqual(Node(0).ancestor_node_at_level(0),
                         Node(0))
        self.assertEqual(Node(0).ancestor_node_at_level(1),
                         None)
        self.assertEqual(Node(1).ancestor_node_at_level(0),
                         Node(0))
        self.assertEqual(Node(1).ancestor_node_at_level(1),
                         Node(1))
        self.assertEqual(Node(1).ancestor_node_at_level(2),
                         None)
        self.assertEqual(Node(4).ancestor_node_at_level(0),
                         Node(0))
        self.assertEqual(Node(4).ancestor_node_at_level(1),
                         Node(1))
        self.assertEqual(Node(4).ancestor_node_at_level(2),
                         Node(4))
        self.assertEqual(Node(4).ancestor_node_at_level(3),
                         None)

    def test_path_to_root(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(list(int(n) for n in Node(0).bucket_path_to_root()),
                         list(reversed([0])))
        self.assertEqual(list(int(n) for n in Node(7).bucket_path_to_root()),
                         list(reversed([0, 1, 3, 7])))
        self.assertEqual(list(int(n) for n in Node(8).bucket_path_to_root()),
                         list(reversed([0, 1, 3, 8])))
        self.assertEqual(list(int(n) for n in Node(9).bucket_path_to_root()),
                         list(reversed([0, 1, 4, 9])))
        self.assertEqual(list(int(n) for n in Node(10).bucket_path_to_root()),
                         list(reversed([0, 1, 4, 10])))
        self.assertEqual(list(int(n) for n in Node(11).bucket_path_to_root()),
                         list(reversed([0, 2, 5, 11])))
        self.assertEqual(list(int(n) for n in Node(12).bucket_path_to_root()),
                         list(reversed([0, 2, 5, 12])))
        self.assertEqual(list(int(n) for n in Node(13).bucket_path_to_root()),
                         list(reversed([0, 2, 6, 13])))
        self.assertEqual(list(int(n) for n in Node(14).bucket_path_to_root()),
                         list(reversed([0, 2, 6, 14])))

    def test_path_from_root(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(list(int(n) for n in Node(0).bucket_path_from_root()),
                         [0])
        self.assertEqual(list(int(n) for n in Node(7).bucket_path_from_root()),
                         [0, 1, 3, 7])
        self.assertEqual(list(int(n) for n in Node(8).bucket_path_from_root()),
                         [0, 1, 3, 8])
        self.assertEqual(list(int(n) for n in Node(9).bucket_path_from_root()),
                         [0, 1, 4, 9])
        self.assertEqual(list(int(n) for n in Node(10).bucket_path_from_root()),
                         [0, 1, 4, 10])
        self.assertEqual(list(int(n) for n in Node(11).bucket_path_from_root()),
                         [0, 2, 5, 11])
        self.assertEqual(list(int(n) for n in Node(12).bucket_path_from_root()),
                         [0, 2, 5, 12])
        self.assertEqual(list(int(n) for n in Node(13).bucket_path_from_root()),
                         [0, 2, 6, 13])
        self.assertEqual(list(int(n) for n in Node(14).bucket_path_from_root()),
                         [0, 2, 6, 14])

    def test_bucket_path_to_root(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(list(Node(0).bucket_path_to_root()),
                         list(reversed([0])))
        self.assertEqual(list(Node(7).bucket_path_to_root()),
                         list(reversed([0, 1, 3, 7])))
        self.assertEqual(list(Node(8).bucket_path_to_root()),
                         list(reversed([0, 1, 3, 8])))
        self.assertEqual(list(Node(9).bucket_path_to_root()),
                         list(reversed([0, 1, 4, 9])))
        self.assertEqual(list(Node(10).bucket_path_to_root()),
                         list(reversed([0, 1, 4, 10])))
        self.assertEqual(list(Node(11).bucket_path_to_root()),
                         list(reversed([0, 2, 5, 11])))
        self.assertEqual(list(Node(12).bucket_path_to_root()),
                         list(reversed([0, 2, 5, 12])))
        self.assertEqual(list(Node(13).bucket_path_to_root()),
                         list(reversed([0, 2, 6, 13])))
        self.assertEqual(list(Node(14).bucket_path_to_root()),
                         list(reversed([0, 2, 6, 14])))

    def test_bucket_path_from_root(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(Node(0).bucket_path_from_root(),
                         [0])
        self.assertEqual(Node(7).bucket_path_from_root(),
                         [0, 1, 3, 7])
        self.assertEqual(Node(8).bucket_path_from_root(),
                         [0, 1, 3, 8])
        self.assertEqual(Node(9).bucket_path_from_root(),
                         [0, 1, 4, 9])
        self.assertEqual(Node(10).bucket_path_from_root(),
                         [0, 1, 4, 10])
        self.assertEqual(Node(11).bucket_path_from_root(),
                         [0, 2, 5, 11])
        self.assertEqual(Node(12).bucket_path_from_root(),
                         [0, 2, 5, 12])
        self.assertEqual(Node(13).bucket_path_from_root(),
                         [0, 2, 6, 13])
        self.assertEqual(Node(14).bucket_path_from_root(),
                         [0, 2, 6, 14])

    def test_repr(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(
            repr(Node(0)),
            "VirtualHeapNode(k=2, bucket=0, level=0, label='')")
        self.assertEqual(
            repr(Node(7)),
            "VirtualHeapNode(k=2, bucket=7, level=3, label='000')")

        Node = VirtualHeap(3).Node
        self.assertEqual(
            repr(Node(0)),
            "VirtualHeapNode(k=3, bucket=0, level=0, label='')")
        self.assertEqual(
            repr(Node(7)),
            "VirtualHeapNode(k=3, bucket=7, level=2, label='10')")

        Node = VirtualHeap(5).Node
        self.assertEqual(
            repr(Node(25)),
            "VirtualHeapNode(k=5, bucket=25, level=2, label='34')")

        Node = VirtualHeap(max_k_labeled).Node
        self.assertEqual(
            repr(Node(0)),
            ("VirtualHeapNode(k=%d, bucket=0, level=0, label='')"
             % (max_k_labeled)))
        self.assertEqual(max_k_labeled >= 2, True)
        self.assertEqual(
            repr(Node(1)),
            ("VirtualHeapNode(k=%d, bucket=1, level=1, label='0')"
             % (max_k_labeled)))

        Node = VirtualHeap(max_k_labeled+1).Node
        self.assertEqual(
            repr(Node(0)),
            ("VirtualHeapNode(k=%d, bucket=0, level=0, label='')"
             % (max_k_labeled+1)))
        self.assertEqual(
            repr(Node(1)),
            ("VirtualHeapNode(k=%d, bucket=1, level=1, label='<unknown>')"
             % (max_k_labeled+1)))
        self.assertEqual(
            repr(Node(max_k_labeled+1)),
            ("VirtualHeapNode(k=%d, bucket=%d, level=1, label='<unknown>')"
             % (max_k_labeled+1,
                max_k_labeled+1)))
        self.assertEqual(
            repr(Node(max_k_labeled+2)),
            ("VirtualHeapNode(k=%d, bucket=%d, level=2, label='<unknown>')"
             % (max_k_labeled+1,
                max_k_labeled+2)))

    def test_str(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(
            str(Node(0)),
            "(0, 0)")
        self.assertEqual(
            str(Node(7)),
            "(3, 0)")

        Node = VirtualHeap(3).Node
        self.assertEqual(
            str(Node(0)),
            "(0, 0)")
        self.assertEqual(
            str(Node(7)),
            "(2, 3)")

        Node = VirtualHeap(5).Node
        self.assertEqual(
            str(Node(25)),
            "(2, 19)")

    def test_label(self):

        Node = VirtualHeap(2).Node
        self.assertEqual(Node(0).label(), "")
        self.assertEqual(Node(1).label(), "0")
        self.assertEqual(Node(2).label(), "1")
        self.assertEqual(Node(3).label(), "00")
        self.assertEqual(Node(4).label(), "01")
        self.assertEqual(Node(5).label(), "10")
        self.assertEqual(Node(6).label(), "11")
        self.assertEqual(Node(7).label(), "000")
        self.assertEqual(Node(8).label(), "001")
        self.assertEqual(Node(9).label(), "010")
        self.assertEqual(Node(10).label(), "011")
        self.assertEqual(Node(11).label(), "100")
        self.assertEqual(Node(12).label(), "101")
        self.assertEqual(Node(13).label(), "110")
        self.assertEqual(Node(14).label(), "111")
        self.assertEqual(Node(15).label(), "0000")
        self.assertEqual(Node(30).label(), "1111")

        for k in _test_labeled_bases:
            Node = VirtualHeap(k).Node
            for b in xrange(calculate_bucket_count_in_heap_with_height(k, 2)+1):
                label = Node(b).label()
                level = Node(b).level
                if label == "":
                    self.assertEqual(b, 0)
                else:
                    self.assertEqual(
                        b,
                        basek_string_to_base10_integer(k, label) + \
                        calculate_bucket_count_in_heap_with_height(k, level-1))

    def test_is_node_on_path(self):
        Node = VirtualHeap(2).Node
        self.assertEqual(
            Node(0).is_node_on_path(
                Node(0)),
            True)
        self.assertEqual(
            Node(0).is_node_on_path(
                Node(1)),
            False)
        self.assertEqual(
            Node(0).is_node_on_path(
                Node(2)),
            False)
        self.assertEqual(
            Node(0).is_node_on_path(
                Node(3)),
            False)

        Node = VirtualHeap(5).Node
        self.assertEqual(
            Node(20).is_node_on_path(
                Node(21)),
            False)
        self.assertEqual(
            Node(21).is_node_on_path(
                Node(4)),
            True)

        Node = VirtualHeap(3).Node
        self.assertEqual(
            Node(7).is_node_on_path(
                Node(0)),
            True)
        self.assertEqual(
            Node(7).is_node_on_path(
                Node(2)),
            True)
        self.assertEqual(
            Node(7).is_node_on_path(
                Node(7)),
            True)
        self.assertEqual(
            Node(7).is_node_on_path(
                Node(8)),
            False)

class TestVirtualHeap(unittest.TestCase):

    def test_init(self):
        vh = VirtualHeap(2, blocks_per_bucket=4)
        self.assertEqual(vh.k, 2)
        self.assertEqual(vh.Node.k, 2)
        self.assertEqual(vh.blocks_per_bucket, 4)
        vh = VirtualHeap(5, blocks_per_bucket=7)
        self.assertEqual(vh.k, 5)
        self.assertEqual(vh.Node.k, 5)
        self.assertEqual(vh.blocks_per_bucket, 7)

    def test_node_label_to_bucket(self):
        vh = VirtualHeap(2)
        self.assertEqual(vh.node_label_to_bucket(""), 0)
        self.assertEqual(vh.node_label_to_bucket("0"), 1)
        self.assertEqual(vh.node_label_to_bucket("1"), 2)
        self.assertEqual(vh.node_label_to_bucket("00"), 3)
        self.assertEqual(vh.node_label_to_bucket("01"), 4)
        self.assertEqual(vh.node_label_to_bucket("10"), 5)
        self.assertEqual(vh.node_label_to_bucket("11"), 6)
        self.assertEqual(vh.node_label_to_bucket("000"), 7)
        self.assertEqual(vh.node_label_to_bucket("001"), 8)
        self.assertEqual(vh.node_label_to_bucket("010"), 9)
        self.assertEqual(vh.node_label_to_bucket("011"), 10)
        self.assertEqual(vh.node_label_to_bucket("100"), 11)
        self.assertEqual(vh.node_label_to_bucket("101"), 12)
        self.assertEqual(vh.node_label_to_bucket("110"), 13)
        self.assertEqual(vh.node_label_to_bucket("111"), 14)
        self.assertEqual(vh.node_label_to_bucket("0000"), 15)
        self.assertEqual(vh.node_label_to_bucket("1111"),
                         calculate_bucket_count_in_heap_with_height(2, 4)-1)

        vh = VirtualHeap(3)
        self.assertEqual(vh.node_label_to_bucket(""), 0)
        self.assertEqual(vh.node_label_to_bucket("0"), 1)
        self.assertEqual(vh.node_label_to_bucket("1"), 2)
        self.assertEqual(vh.node_label_to_bucket("2"), 3)
        self.assertEqual(vh.node_label_to_bucket("00"), 4)
        self.assertEqual(vh.node_label_to_bucket("01"), 5)
        self.assertEqual(vh.node_label_to_bucket("02"), 6)
        self.assertEqual(vh.node_label_to_bucket("10"), 7)
        self.assertEqual(vh.node_label_to_bucket("11"), 8)
        self.assertEqual(vh.node_label_to_bucket("12"), 9)
        self.assertEqual(vh.node_label_to_bucket("20"), 10)
        self.assertEqual(vh.node_label_to_bucket("21"), 11)
        self.assertEqual(vh.node_label_to_bucket("22"), 12)
        self.assertEqual(vh.node_label_to_bucket("000"), 13)
        self.assertEqual(vh.node_label_to_bucket("222"),
                         calculate_bucket_count_in_heap_with_height(3, 3)-1)

        for k in xrange(2, max_k_labeled+1):
            for h in xrange(5):
                vh = VirtualHeap(k)
                largest_symbol = numerals[k-1]
                self.assertEqual(vh.k, k)
                self.assertEqual(vh.node_label_to_bucket(""), 0)
                self.assertEqual(vh.node_label_to_bucket(largest_symbol * h),
                                 calculate_bucket_count_in_heap_with_height(k, h)-1)

    def test_ObjectCountAtLevel(self):
        for k in _test_bases:
            for height in xrange(k+2):
                for blocks_per_bucket in xrange(1, 5):
                    vh = VirtualHeap(k, blocks_per_bucket=blocks_per_bucket)
                    for l in xrange(height+1):
                        cnt = k**l
                        self.assertEqual(vh.bucket_count_at_level(l), cnt)
                        self.assertEqual(vh.node_count_at_level(l), cnt)
                        self.assertEqual(vh.block_count_at_level(l),
                                         cnt * blocks_per_bucket)

    def test_bucket_to_block(self):
        for k in xrange(2, 6):
            for blocks_per_bucket in xrange(1, 5):
                heap = VirtualHeap(k, blocks_per_bucket=blocks_per_bucket)
                for b in xrange(20):
                    self.assertEqual(heap.bucket_to_block(b),
                                     blocks_per_bucket * b)

    def test_node_count_at_level(self):
        self.assertEqual(VirtualHeap(2).node_count_at_level(0), 1)
        self.assertEqual(VirtualHeap(2).node_count_at_level(1), 2)
        self.assertEqual(VirtualHeap(2).node_count_at_level(2), 4)
        self.assertEqual(VirtualHeap(2).node_count_at_level(3), 8)
        self.assertEqual(VirtualHeap(2).node_count_at_level(4), 16)

        self.assertEqual(VirtualHeap(3).node_count_at_level(0), 1)
        self.assertEqual(VirtualHeap(3).node_count_at_level(1), 3)
        self.assertEqual(VirtualHeap(3).node_count_at_level(2), 9)
        self.assertEqual(VirtualHeap(3).node_count_at_level(3), 27)
        self.assertEqual(VirtualHeap(3).node_count_at_level(4), 81)

        self.assertEqual(VirtualHeap(4).node_count_at_level(0), 1)
        self.assertEqual(VirtualHeap(4).node_count_at_level(1), 4)
        self.assertEqual(VirtualHeap(4).node_count_at_level(2), 16)
        self.assertEqual(VirtualHeap(4).node_count_at_level(3), 64)
        self.assertEqual(VirtualHeap(4).node_count_at_level(4), 256)

    def test_first_node_at_level(self):
        self.assertEqual(VirtualHeap(2).first_node_at_level(0), 0)
        self.assertEqual(VirtualHeap(2).first_node_at_level(1), 1)
        self.assertEqual(VirtualHeap(2).first_node_at_level(2), 3)
        self.assertEqual(VirtualHeap(2).first_node_at_level(3), 7)
        self.assertEqual(VirtualHeap(2).first_node_at_level(4), 15)

        self.assertEqual(VirtualHeap(3).first_node_at_level(0), 0)
        self.assertEqual(VirtualHeap(3).first_node_at_level(1), 1)
        self.assertEqual(VirtualHeap(3).first_node_at_level(2), 4)
        self.assertEqual(VirtualHeap(3).first_node_at_level(3), 13)
        self.assertEqual(VirtualHeap(3).first_node_at_level(4), 40)

        self.assertEqual(VirtualHeap(4).first_node_at_level(0), 0)
        self.assertEqual(VirtualHeap(4).first_node_at_level(1), 1)
        self.assertEqual(VirtualHeap(4).first_node_at_level(2), 5)
        self.assertEqual(VirtualHeap(4).first_node_at_level(3), 21)
        self.assertEqual(VirtualHeap(4).first_node_at_level(4), 85)

    def test_last_node_at_level(self):
        self.assertEqual(VirtualHeap(2).last_node_at_level(0), 0)
        self.assertEqual(VirtualHeap(2).last_node_at_level(1), 2)
        self.assertEqual(VirtualHeap(2).last_node_at_level(2), 6)
        self.assertEqual(VirtualHeap(2).last_node_at_level(3), 14)
        self.assertEqual(VirtualHeap(2).last_node_at_level(4), 30)

        self.assertEqual(VirtualHeap(3).last_node_at_level(0), 0)
        self.assertEqual(VirtualHeap(3).last_node_at_level(1), 3)
        self.assertEqual(VirtualHeap(3).last_node_at_level(2), 12)
        self.assertEqual(VirtualHeap(3).last_node_at_level(3), 39)
        self.assertEqual(VirtualHeap(3).last_node_at_level(4), 120)

        self.assertEqual(VirtualHeap(4).last_node_at_level(0), 0)
        self.assertEqual(VirtualHeap(4).last_node_at_level(1), 4)
        self.assertEqual(VirtualHeap(4).last_node_at_level(2), 20)
        self.assertEqual(VirtualHeap(4).last_node_at_level(3), 84)
        self.assertEqual(VirtualHeap(4).last_node_at_level(4), 340)

    def test_random_node_up_to_level(self):
        for k in xrange(2,6):
            heap = VirtualHeap(k)
            for l in xrange(4):
                for t in xrange(2 * calculate_bucket_count_in_heap_with_height(k, l)):
                    node = heap.random_node_up_to_level(l)
                    self.assertEqual(node.level <= l, True)

    def test_random_node_at_level(self):
        for k in xrange(2,6):
            heap = VirtualHeap(k)
            for l in xrange(4):
                for t in xrange(2 * calculate_bucket_count_in_heap_at_level(k, l)):
                    node = heap.random_node_at_level(l)
                    self.assertEqual(node.level == l, True)

    def test_first_block_at_level(self):
        for blocks_per_bucket in xrange(1, 5):
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(0), 0 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(1), 1 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(2), 3 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(3), 7 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(4), 15 * blocks_per_bucket)

            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(0), 0 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(1), 1 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(2), 4 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(3), 13 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(4), 40 * blocks_per_bucket)

            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(0), 0 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(1), 1 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(2), 5 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(3), 21 * blocks_per_bucket)
            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             first_block_at_level(4), 85 * blocks_per_bucket)

    def test_last_block_at_level(self):
        for blocks_per_bucket in xrange(1, 5):
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(0), 0 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(1), 2 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(2), 6 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(3), 14 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(2, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(4), 30 * blocks_per_bucket + (blocks_per_bucket-1))

            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(0), 0 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(1), 3 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(2), 12 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(3), 39 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(3, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(4), 120 * blocks_per_bucket + (blocks_per_bucket-1))

            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(0), 0 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(1), 4 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(2), 20 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(3), 84 * blocks_per_bucket + (blocks_per_bucket-1))
            self.assertEqual(VirtualHeap(4, blocks_per_bucket=blocks_per_bucket).\
                             last_block_at_level(4), 340 * blocks_per_bucket + (blocks_per_bucket-1))

    def test_block_to_bucket(self):
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=1).block_to_bucket(0), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=1).block_to_bucket(1), 1)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=1).block_to_bucket(2), 2)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=1).block_to_bucket(3), 3)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=1).block_to_bucket(4), 4)

        self.assertEqual(VirtualHeap(2, blocks_per_bucket=2).block_to_bucket(0), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=2).block_to_bucket(1), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=2).block_to_bucket(2), 1)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=2).block_to_bucket(3), 1)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=2).block_to_bucket(4), 2)

        self.assertEqual(VirtualHeap(2, blocks_per_bucket=3).block_to_bucket(0), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=3).block_to_bucket(1), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=3).block_to_bucket(2), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=3).block_to_bucket(3), 1)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=3).block_to_bucket(4), 1)

        self.assertEqual(VirtualHeap(2, blocks_per_bucket=4).block_to_bucket(0), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=4).block_to_bucket(1), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=4).block_to_bucket(2), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=4).block_to_bucket(3), 0)
        self.assertEqual(VirtualHeap(2, blocks_per_bucket=4).block_to_bucket(4), 1)

    def test_root_node(self):
        for k in range(2, 6):
            for blocks_per_bucket in range(1, 5):
                heap = VirtualHeap(k, blocks_per_bucket=blocks_per_bucket)
                root = heap.root_node()
                self.assertEqual(root, 0)
                self.assertEqual(root.bucket, 0)
                self.assertEqual(root.level, 0)
                self.assertEqual(root.parent_node(), None)

class TestSizedVirtualHeap(unittest.TestCase):

    def test_init(self):
        vh = SizedVirtualHeap(2, 8, blocks_per_bucket=4)
        self.assertEqual(vh.k, 2)
        self.assertEqual(vh.Node.k, 2)
        self.assertEqual(vh.blocks_per_bucket, 4)
        vh = SizedVirtualHeap(5, 9, blocks_per_bucket=7)
        self.assertEqual(vh.k, 5)
        self.assertEqual(vh.Node.k, 5)
        self.assertEqual(vh.blocks_per_bucket, 7)

    def test_height(self):
        vh = SizedVirtualHeap(2, 3, blocks_per_bucket=4)
        self.assertEqual(vh.height, 3)
        vh = SizedVirtualHeap(5, 6, blocks_per_bucket=7)
        self.assertEqual(vh.height, 6)

    def test_levels(self):
        vh = SizedVirtualHeap(2, 3, blocks_per_bucket=4)
        self.assertEqual(vh.levels, 4)
        vh = SizedVirtualHeap(5, 6, blocks_per_bucket=7)
        self.assertEqual(vh.levels, 7)

    def test_first_level(self):
        vh = SizedVirtualHeap(2, 3, blocks_per_bucket=4)
        self.assertEqual(vh.first_level, 0)
        vh = SizedVirtualHeap(5, 6, blocks_per_bucket=7)
        self.assertEqual(vh.first_level, 0)

    def test_last_level(self):
        vh = SizedVirtualHeap(2, 3, blocks_per_bucket=4)
        self.assertEqual(vh.last_level, 3)
        self.assertEqual(vh.last_level, vh.levels-1)
        self.assertEqual(vh.last_level, vh.height)
        vh = SizedVirtualHeap(5, 6, blocks_per_bucket=7)
        self.assertEqual(vh.last_level, 6)
        self.assertEqual(vh.last_level, vh.levels-1)
        self.assertEqual(vh.last_level, vh.height)

    def test_ObjectCount(self):
        for k in _test_bases:
            for height in xrange(k+2):
                for blocks_per_bucket in xrange(1, 5):
                    vh = SizedVirtualHeap(k,
                                          height,
                                          blocks_per_bucket=blocks_per_bucket)
                    cnt = (((k**(height+1))-1)//(k-1))
                    self.assertEqual(vh.bucket_count(), cnt)
                    self.assertEqual(vh.node_count(), cnt)
                    self.assertEqual(vh.block_count(), cnt * blocks_per_bucket)

    def test_LeafObjectCount(self):
        for k in _test_bases:
            for height in xrange(k+2):
                for blocks_per_bucket in xrange(1, 5):
                    vh = SizedVirtualHeap(k,
                                          height,
                                          blocks_per_bucket=blocks_per_bucket)
                    self.assertEqual(vh.leaf_bucket_count(),
                                     vh.bucket_count_at_level(vh.height))
                    self.assertEqual(vh.leaf_node_count(),
                                     vh.node_count_at_level(vh.height))
                    self.assertEqual(vh.leaf_block_count(),
                                     vh.block_count_at_level(vh.height))


    def test_FirstLeafObject(self):
        vh = SizedVirtualHeap(2, 3, blocks_per_bucket=3)
        self.assertEqual(vh.first_leaf_node(), 7)
        self.assertEqual(vh.first_leaf_block(), 7*3)

    def test_LastLeafObject(self):
        vh = SizedVirtualHeap(2, 3, blocks_per_bucket=3)
        self.assertEqual(vh.last_leaf_node(), 14)
        self.assertEqual(vh.last_leaf_block(), 14*3 + 2)

    def test_random_node(self):
        for k in xrange(2,6):
            height = 3
            heap = SizedVirtualHeap(k, height)
            for t in xrange(2 * heap.bucket_count()):
                node = heap.random_node()
                self.assertEqual(0 <= node.level <= height, True)

    def test_random_leaf_node(self):
        for k in xrange(2,6):
            height = 3
            heap = SizedVirtualHeap(k, height)
            for t in xrange(2 * heap.bucket_count()):
                node = heap.random_leaf_node()
                self.assertEqual(node.level, height)

    def _assert_file_equals_baselines(self, fname, bname):
        with open(fname)as f:
            flines = f.readlines()
        with open(bname) as f:
            blines = f.readlines()
        self.assertListEqual(flines, blines)
        os.remove(fname)

    def test_write_as_dot(self):

        for k, h, b, maxl in [(2, 3, 1, None),
                              (2, 3, 2, None),
                              (3, 3, 1, None),
                              (3, 3, 2, None),
                              (3, 10, 2, 4),
                              (200, 0, 1, None)]:
            if maxl is None:
                label = "k%d_h%d_b%d" % (k, h, b)
            else:
                label = "k%d_h%d_b%d" % (k, maxl-1, b)
            heap = SizedVirtualHeap(k, h, blocks_per_bucket=b)

            fname = label+".dot"
            with open(os.path.join(thisdir, fname), "w") as f:
                heap.write_as_dot(f, max_levels=maxl)
            self._assert_file_equals_baselines(
                os.path.join(thisdir, fname),
                os.path.join(baselinedir, fname))

            data = list(range(heap.block_count()))
            fname = label+"_data.dot"
            with open(os.path.join(thisdir, fname), "w") as f:
                heap.write_as_dot(f, data=data, max_levels=maxl)
            self._assert_file_equals_baselines(
                os.path.join(thisdir, fname),
                os.path.join(baselinedir, fname))

    def test_save_image_as_pdf(self):

        for k, h, b, maxl in [(2, 3, 1, None),
                              (2, 3, 2, None),
                              (3, 3, 1, None),
                              (3, 3, 2, None),
                              (3, 10, 2, 4)]:
            if maxl is None:
                label = "k%d_h%d_b%d" % (k, h, b)
            else:
                label = "k%d_h%d_b%d" % (k, maxl-1, b)
            heap = SizedVirtualHeap(k, h, blocks_per_bucket=b)

            fname = label+".pdf"
            try:
                os.remove(os.path.join(thisdir, fname))
            except OSError:                            # pragma: no cover
                pass                                   # pragma: no cover

            rc = heap.save_image_as_pdf(os.path.join(thisdir, label),
                                        max_levels=maxl)

            if not has_dot:
                self.assertEqual(rc, False)
            else:
                self.assertEqual(rc, True)
                self.assertEqual(
                    os.path.exists(os.path.join(thisdir, fname)), True)
                try:
                    os.remove(os.path.join(thisdir, fname))
                except OSError:                        # pragma: no cover
                    pass                               # pragma: no cover

            data = list(range(heap.block_count()))
            fname = label+"_data.pdf"
            try:
                os.remove(os.path.join(thisdir, fname))
            except OSError:                            # pragma: no cover
                pass                                   # pragma: no cover
            rc = heap.save_image_as_pdf(os.path.join(thisdir, fname),
                                        data=data,
                                        max_levels=maxl)
            if not has_dot:
                self.assertEqual(rc, False)
            else:
                self.assertEqual(rc, True)
                self.assertEqual(
                    os.path.exists(os.path.join(thisdir, fname)), True)
                try:
                    os.remove(os.path.join(thisdir, fname))
                except OSError:                        # pragma: no cover
                    pass                               # pragma: no cover

class TestMisc(unittest.TestCase):

    def test_calculate_bucket_level(self):
        self.assertEqual(calculate_bucket_level(2, 0), 0)
        self.assertEqual(calculate_bucket_level(2, 1), 1)
        self.assertEqual(calculate_bucket_level(2, 2), 1)
        self.assertEqual(calculate_bucket_level(2, 3), 2)
        self.assertEqual(calculate_bucket_level(2, 4), 2)
        self.assertEqual(calculate_bucket_level(2, 5), 2)
        self.assertEqual(calculate_bucket_level(2, 6), 2)
        self.assertEqual(calculate_bucket_level(2, 7), 3)

        self.assertEqual(calculate_bucket_level(3, 0), 0)
        self.assertEqual(calculate_bucket_level(3, 1), 1)
        self.assertEqual(calculate_bucket_level(3, 2), 1)
        self.assertEqual(calculate_bucket_level(3, 3), 1)
        self.assertEqual(calculate_bucket_level(3, 4), 2)
        self.assertEqual(calculate_bucket_level(3, 5), 2)
        self.assertEqual(calculate_bucket_level(3, 6), 2)
        self.assertEqual(calculate_bucket_level(3, 7), 2)
        self.assertEqual(calculate_bucket_level(3, 8), 2)
        self.assertEqual(calculate_bucket_level(3, 9), 2)
        self.assertEqual(calculate_bucket_level(3, 10), 2)
        self.assertEqual(calculate_bucket_level(3, 11), 2)
        self.assertEqual(calculate_bucket_level(3, 12), 2)
        self.assertEqual(calculate_bucket_level(3, 13), 3)

        self.assertEqual(calculate_bucket_level(4, 0), 0)
        self.assertEqual(calculate_bucket_level(4, 1), 1)
        self.assertEqual(calculate_bucket_level(4, 2), 1)
        self.assertEqual(calculate_bucket_level(4, 3), 1)
        self.assertEqual(calculate_bucket_level(4, 4), 1)

        self.assertEqual(calculate_bucket_level(4, 5), 2)
        self.assertEqual(calculate_bucket_level(4, 6), 2)
        self.assertEqual(calculate_bucket_level(4, 7), 2)
        self.assertEqual(calculate_bucket_level(4, 8), 2)

        self.assertEqual(calculate_bucket_level(4, 9), 2)
        self.assertEqual(calculate_bucket_level(4, 10), 2)
        self.assertEqual(calculate_bucket_level(4, 11), 2)
        self.assertEqual(calculate_bucket_level(4, 12), 2)

        self.assertEqual(calculate_bucket_level(4, 13), 2)
        self.assertEqual(calculate_bucket_level(4, 14), 2)
        self.assertEqual(calculate_bucket_level(4, 15), 2)
        self.assertEqual(calculate_bucket_level(4, 16), 2)

        self.assertEqual(calculate_bucket_level(4, 17), 2)
        self.assertEqual(calculate_bucket_level(4, 18), 2)
        self.assertEqual(calculate_bucket_level(4, 19), 2)
        self.assertEqual(calculate_bucket_level(4, 20), 2)

        self.assertEqual(calculate_bucket_level(4, 21), 3)

    def test_clib_calculate_bucket_level(self):
        for k in _test_bases:
            for b in xrange(calculate_bucket_count_in_heap_with_height(k, 2)+2):
                self.assertEqual(_clib.calculate_bucket_level(k, b),
                                 calculate_bucket_level(k, b))
        for k, b in [(89, 14648774),
                     (89, 14648775),
                     (90, 14648774),
                     (90, 14648775)]:
            self.assertEqual(_clib.calculate_bucket_level(k, b),
                             calculate_bucket_level(k, b))

    def test_clib_calculate_last_common_level(self):
        for k in range(2, 8):
            for b1 in xrange(calculate_bucket_count_in_heap_with_height(k, 2)+2):
                for b2 in xrange(calculate_bucket_count_in_heap_with_height(k, 2)+2):
                    self.assertEqual(_clib.calculate_last_common_level(k, b1, b2),
                                     calculate_last_common_level(k, b1, b2))
        for k in [89,90]:
            for b1 in [0, 100, 10000, 14648774, 14648775]:
                for b2 in [0, 100, 10000, 14648774, 14648775]:
                    self.assertEqual(_clib.calculate_last_common_level(k, b1, b2),
                                     calculate_last_common_level(k, b1, b2))

    def test_calculate_necessary_heap_height(self):
        self.assertEqual(calculate_necessary_heap_height(2, 1), 0)
        self.assertEqual(calculate_necessary_heap_height(2, 2), 1)
        self.assertEqual(calculate_necessary_heap_height(2, 3), 1)
        self.assertEqual(calculate_necessary_heap_height(2, 4), 2)
        self.assertEqual(calculate_necessary_heap_height(2, 5), 2)
        self.assertEqual(calculate_necessary_heap_height(2, 6), 2)
        self.assertEqual(calculate_necessary_heap_height(2, 7), 2)
        self.assertEqual(calculate_necessary_heap_height(2, 8), 3)

        self.assertEqual(calculate_necessary_heap_height(3, 1), 0)
        self.assertEqual(calculate_necessary_heap_height(3, 2), 1)
        self.assertEqual(calculate_necessary_heap_height(3, 3), 1)
        self.assertEqual(calculate_necessary_heap_height(3, 4), 1)
        self.assertEqual(calculate_necessary_heap_height(3, 5), 2)
        self.assertEqual(calculate_necessary_heap_height(3, 6), 2)
        self.assertEqual(calculate_necessary_heap_height(3, 7), 2)
        self.assertEqual(calculate_necessary_heap_height(3, 8), 2)
        self.assertEqual(calculate_necessary_heap_height(3, 9), 2)
        self.assertEqual(calculate_necessary_heap_height(3, 10), 2)
        self.assertEqual(calculate_necessary_heap_height(3, 11), 2)
        self.assertEqual(calculate_necessary_heap_height(3, 12), 2)
        self.assertEqual(calculate_necessary_heap_height(3, 13), 2)
        self.assertEqual(calculate_necessary_heap_height(3, 14), 3)
        self.assertEqual(calculate_necessary_heap_height(3, 15), 3)
        self.assertEqual(calculate_necessary_heap_height(3, 16), 3)

if __name__ == "__main__":
    unittest.main()                                    # pragma: no cover
