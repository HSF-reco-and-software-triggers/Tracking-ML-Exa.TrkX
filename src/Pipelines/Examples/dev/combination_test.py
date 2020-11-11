import itertools
import yaml

with open('train_gnn.yaml', 'r') as stream:
    try:
        inputdict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

total_list = {k: (v if type(v) == list else [v]) for (k,v) in inputdict.items()}
keys, values = zip(*total_list.items())

# Build list of config dictionaries
configs = []
[configs.append(dict(zip(keys, bundle))) for bundle in product(*values)];