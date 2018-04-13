# lcj_MNIST

A slight change from https://github.com/peikexin9/deepxplore/tree/master/MNIST

Generate images with shape (1,28,28,1) to activate a set of neurons (instead of an individual neuron as in current implementation).

# What I changed?

> + init_coverage_tables --> init_set_coverage_tables
> + init_dict --> init_set_dict
> + neuron_to_cover --> set_to_cover
> + neuron_covered --> set_covered
> + update_coverage --> set_update_coverage

# Change about loss
```python
# loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
loss1_neuron = K.mean(K.stack([K.mean(model1.get_layer(location[0]).output[..., location[1]]) for location in location_list1]))
```

# How to use

```Bash
python gen_diff.py -h
```

An example:

```Bash
python gen_diff.py 'occl' 0.5 0.1 20 20 30 0
```

