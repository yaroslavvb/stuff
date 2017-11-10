#!/usr/bin/env python

import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='script to fill EFS with data')

parser.add_argument('--gb', type=int, default=100, metavar='N',
                    help='how many GBs to dump')
parser.add_argument('--chunk_gb', type=int, default=1, metavar='N',
                    help='how many GBs to dump')
parser.add_argument('--fn', type=str, default="fill", metavar='N',
                    help='filename')
args = parser.parse_args()

def main():
  chunk_size = args.chunk_gb*1e9
  current_size = 0

  file_counter = 0
  max_file_counter = int(math.ceil(args.gb/args.chunk_gb))
  while current_size < args.gb*1e9:
    fn = args.fn+"-%05d-of-%05d"%(file_counter, max_file_counter)
    file_counter+=1
    with open(fn, 'wb') as out:
      out.write(np.random.bytes(chunk_size))
    print("Wrote %5.1f GBs"%(current_size/1e9))
    current_size+=chunk_size

if __name__=='__main__':
  main()
