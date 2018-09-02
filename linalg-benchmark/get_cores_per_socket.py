#!/usr/bin/env python
"""Simple script to parse cpuinfo and generate command to limit to a single physical socket"""

import re
socket_re = re.compile(".*?processor.*?(?P<cpu>\d+).*?physical id.*?(?P<socket>\d+).*?power", flags=re.S)
from collections import defaultdict
socket_dict = defaultdict(list)
for cpu, socket in socket_re.findall(open('/proc/cpuinfo').read()):
  socket_dict[socket].append(cpu)


for socket,cpus in socket_dict.items():
  print('to set to socket', socket)
  print('export GOMP_CPU_AFFINITY=%s'%(','.join(cpus)))
