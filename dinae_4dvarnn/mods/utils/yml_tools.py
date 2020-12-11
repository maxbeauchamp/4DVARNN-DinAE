from dinae_4dvarnn import *

config_file = Path('config.yaml')

yaml = ruamel.yaml.YAML(typ='safe')


@yaml.register_class
class Array2d:
    yaml_tag = '!2darray'
    @classmethod
    def from_yaml(cls, constructor, node):
        array = constructor.construct_sequence(node, deep=True)
        return np.array(array)

@yaml.register_class
class Array1d:
    yaml_tag = '!1darray'
    @classmethod
    def from_yaml(cls, constructor, node):
        array = constructor.construct_sequence(node, deep=True)
        return  np.reshape(np.array(array),np.array(array).size)
