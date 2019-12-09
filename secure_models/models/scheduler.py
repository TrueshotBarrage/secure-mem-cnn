import random
import numpy as np

def get_jobs(x, y, i, j, num_blocks):
   tasks = set()
   for n in range(x, x + i):
      for m in range(y, y + j):
         if (n < num_blocks and m < num_blocks):
            tasks.add((n, m))

   return tasks
      
def get_bias_jobs(y, j, num_blocks):
   tasks = set()
   for m in range(y, y + j):
      if (m < num_blocks):
         tasks.add((-1, m))

   return tasks

def feasible_schedule(num_blocks, max_blocks, max_kernels = None):
   table = np.ones((num_blocks, num_blocks))

   job_num = 0
   jobs = dict()

   i = max_blocks / 2
   j = max_blocks - i
   if (max_kernels is not None):
      if (i * j > max_kernels):
         i = max_kernels / 2
         j = max_kernels - i

   x = 0
   y = 0
   while (x < num_blocks):
      while (y < num_blocks):
         jobs[job_num] = get_jobs(x, y, i, j, num_blocks)
         job_num += 1
         y += j
      x += i    
      y = 0     

   # add bias tasks
   j = min(max_kernels, max_blocks)
   while (y < num_blocks):
      jobs[job_num] = get_bias_jobs(y, j, num_blocks)
      y += j
      job_num += 1
      

   return (job_num, jobs)          

def schedule(num_blocks, max_blocks, max_tasks = None):
   # num_blocks is the number of blocks each of the feature maps is divided into
   # max_blocks is the maximum number of blocks that can read in for each task
   # max_tasks is the maximum number of tasks that can be scheduled at once
   # returns a set of scheduling jobs
   
   num_jobs, jobs = feasible_schedule(num_blocks, max_blocks, max_tasks)
   order = np.arange(num_jobs)
   random.shuffle(order)
   return jobs, order   
