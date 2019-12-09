import cffi

#
# C functions that speed up commonly
# executed heap calculations in tree-based
# orams
#

ffi = cffi.FFI()
ffi.cdef(
"""
int calculate_bucket_level(unsigned int k,
                           unsigned long long b);
int calculate_last_common_level(unsigned int k,
                                unsigned long long b1,
                                unsigned long long b2);
""")

ffi.set_source("pyoram.util._virtual_heap_helper",
"""
#include <stdio.h>
#include <stdlib.h>

int calculate_bucket_level(unsigned int k,
                           unsigned long long b)
{
   unsigned int h;
   unsigned long long pow;
   if (k == 2) {
      // This is simply log2floor(b+1)
      h = 0;
      b += 1;
      while (b >>= 1) {++h;}
      return h;
   }
   b = (k - 1) * (b + 1) + 1;
   h = 0;
   pow = k;
   while (pow < b) {++h; pow *= k;}
   return h;
}

int calculate_last_common_level(unsigned int k,
                                unsigned long long b1,
                                unsigned long long b2)
{
   int level1, level2;
   level1 = calculate_bucket_level(k, b1);
   level2 = calculate_bucket_level(k, b2);
   if (level1 != level2) {
      if (level1 > level2) {
         while (level1 != level2) {
            b1 = (b1 - 1)/k;
            --level1;
         }
      }
      else {
         while (level2 != level1) {
            b2 = (b2 - 1)/k;
            --level2;
         }
      }
   }
   while (b1 != b2) {
      b1 = (b1 - 1)/k;
      b2 = (b2 - 1)/k;
      --level1;
   }
   return level1;
}
""")

if __name__ == "__main__":
    ffi.compile()
