from command_builder import *
from pprint import pprint as pp
import yaml

def main():

  
  with open('configs/aws/yaroslav.yaml') as stream:
    config_yaml = yaml.load(stream)

  configs = LoadYamlRunConfig(config_yaml, 1)
#  pp(configs)

  config = configs[0]

  worker_hosts = ['1','2']
  worker_hosts_str = ','.join(worker_hosts)
  ps_hosts = ['a','b']
  ps_hosts_str = ','.join(ps_hosts)
  for i,worker in enumerate(worker_hosts):
    print(BuildDistributedCommandWorker(config, worker_hosts_str, ps_hosts_str, i))
    
  for i,worker in enumerate(ps_hosts):
    print(BuildDistributedCommandPS(config, worker_hosts_str, ps_hosts_str, i))
                                
  

  

if __name__=='__main__':
  main()
