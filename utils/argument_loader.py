import yaml
import argparse
import os

class ArgumentLoader(object):
    """
    Loading yaml configs file
    """
    def __init__(self, config_type='yaml'):
        super(ArgumentLoader, self).__init__()
        self.config_type = config_type
        self.argparser = argparse.ArgumentParser('Yaml configs loader')

    def call(self, *args, **kwargs):
        self.argparser.add_argument('-c','--config-file')
        path = os.path.abspath(self.argparser.parse_args().config_file)
        configs = yaml.safe_load(open(path))
        return configs
    
def main():
    t = ArgumentLoader()
    configs = t.call()
    print(configs)
    
if __name__ == '__main__':
    main()