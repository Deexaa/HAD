import os
import importlib

root_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = os.path.join(root_dir, 'methods')

def get_all_models():
    return [model.split('.')[0] for model in os.listdir(root_dir)
        if not model.find('__') > -1 and 'py' in model]

names = {}
for model in get_all_models():
    mod = importlib.import_module('methods.' + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)

def get_model(args):
    return names[args.method](args)

