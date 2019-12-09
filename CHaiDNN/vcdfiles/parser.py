import sys
import numpy as np
import matplotlib.pyplot as plt

from Verilog_VCD import *

def get_signal_name(signal):
   signal = signal[i].split('.')[-1]
   return signal.split('[')[0]

def get_address_signals(file):
   signals = list_sigs(file)
   address_signals = []
   for signal in signals:
      if ('datain' in signal or 'dataout' in signal):
#         if ('weight' in signal or 'wts' in signal or 'scalar_conv_args' in signal):
            address_signals.append(signal)
   return address_signals   

def parse_data(signal):
   signal_data = signal['tv']
   times = np.empty((0))
   values = np.empty((0))
   for (t, v) in signal_data:
      try:
         w = int(v, 2)
         signal_name = signal['nets'][0]['name']
         times = np.append(times, t)
         values = np.append(values, w)
      except:
         pass

   signal_name = signal['nets'][0]['name']
   return (signal_name, times, values)
   
def plot_signals(data, out_dir):
   for signal in data.values():
      signal_name, times, values = parse_data(signal)
      
      if (len(times) != 0):
         plt.figure()
         plt.plot(times, values, 'bx')
         plt.yscale('log')
         plt.xlabel('Time')
         plt.ylabel('Address')
         plt.title('Memory Access vs. Time: Signal ' + signal_name)
         plt.grid(True)
         out_path = out_dir + '/' + signal_name + '.png'
         plt.savefig(out_path)
         plt.close()     
      
def plot_all_signals(data, out_dir):
   plt.figure()
   ax = plt.subplot(111)

   i = j = k = l = 0
   for signal in data.values():
      signal_name, times, values = parse_data(signal)
      if ('scalar_conv_args_datain' in signal_name):
         print(signal_name)
         np.savez(out_dir + '/' + signal_name, times, values)
         plt.plot(times, values, 'co', label='Conv Args' if i == 0 else '')
         i += 1
      elif ('datain' in signal_name and 'weights' in signal_name):
         print(signal_name)
         np.savez(out_dir + '/' + signal_name, times, values)
         plt.plot(times, values, 'bx', label='Weights' if j == 0 else '')
         j += 1
      elif ('datain' in signal_name and 'input' in signal_name):
         print(signal_name)
         np.savez(out_dir + '/' + signal_name, times, values)
         plt.plot(times, values, 'r+', label='Input' if k == 0 else '')
         k += 1
      elif ('dataout' in signal_name and ('output' in signal_name or 'out_V' in signal_name or 'istg' in signal_name)):
         print(signal_name)
         np.savez(out_dir + '/' + signal_name, times, values)
         plt.plot(times, values, 'g+', label='Output' if l == 0 else '')
         l += 1
      
 
   box = ax.get_position()
   ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
   plt.legend(loc='center left', bbox_to_anchor=[1, 0.5])
   plt.yscale('log')
   plt.xlabel('Time')
   plt.ylabel('Address')
   plt.title('Memory Access vs Time')
   plt.grid(True)
   out_path = out_dir + '/memoryaccesses.png'
   plt.savefig(out_path)
   plt.close()
      
def main():
   file = sys.argv[1]
   out_dir = sys.argv[2]
   signals = get_address_signals(file)
   data = parse_vcd(file, siglist=signals)
   plot_signals(data, out_dir)
   plot_all_signals(data, out_dir)

if __name__ == '__main__':
   main()
