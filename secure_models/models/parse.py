import numpy as np
import sys
import matplotlib.pyplot as plt

out_dir = 'plots'

def main():
   save_path = ''

   plt.figure()
   ax = plt.subplot(111)

   read_count = 0
   write_count = 0

   for i in range(1, len(sys.argv)):
       file_path = sys.argv[i] + '.npz'
       save_path += sys.argv[i] + '_'
       with np.load(file_path) as data:
          read_times = data['read_times']
          read_accesses = data['read_accesses']
          write_times = data['write_times']
          write_accesses = data['write_accesses']
          plt.plot(read_times, read_accesses, 'bx', label = 'Read Accesses')
          plt.plot(write_times, write_accesses, 'r+', label = 'Write Accesses')
          read_count += read_times.size
          write_count += write_times.size
       
   box = ax.get_position()
   ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
   plt.xlabel('Time (s)')
   plt.ylabel('Memory Access (Block #)')
   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   save_path = save_path[:-1]
   out_path = out_dir + '/' + save_path
   plt.savefig(out_path)     

   print('Number of read memory accesses:		' + str(read_count))
   print('Number of write memory accessses: 	' + str(write_count))
   print('Number of total memory accesses:	' + str(read_count + write_count))

if __name__ == '__main__':
   main()
