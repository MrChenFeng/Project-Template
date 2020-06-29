from argparse import Namespace
import argparse
import yaml


def yaml_config():
    parser = argparse.ArgumentParser(description='Please set parameters by xxx.yaml config files.')
    parser.add_argument('-c', '--config', type=str, help='Absolute path of config file.')
    path = parser.parse_args().config
    opts = yaml.safe_load(open(path))
    args = Namespace(**opts)

    return args


# print(yaml_config())
#
#
# opts = yaml.safe_load(open('/home/chen/SESSL/configurations/train.yaml'))
# args = Namespace(**opts)
