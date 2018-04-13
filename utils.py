import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model


# util function to convert a tensor into a valid image
def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3

def init_set_coverage_tables(model1, model2, model3, num_set, set_cap):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    result1 = init_set_dict(model1, model_layer_dict1, num_set, set_cap)
    result2 = init_set_dict(model2, model_layer_dict2, num_set, set_cap)
    result3 = init_set_dict(model3, model_layer_dict3, num_set, set_cap)
    if (result1 and result2 and result3) == False:
        return False
    return model_layer_dict1, model_layer_dict2, model_layer_dict3

def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False

def init_set_dict(model, model_layer_dict, num_set, set_cap):
    set_notfull = range(num_set)
    neurons = []
    # get a list of all neurons
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            neurons.append(str(layer.name)+'_'+str(index))
    # check the capacity of set
    if set_cap > len(neurons)/2:
        return False
    # select neurons for each set
    for i in range(num_set):
        set_neurons = []
        for i in range(set_cap):
            neurons_to_append = random.choice(neurons)
            while (neurons_to_append in set_neurons):
                neurons_to_append = random.choice(neurons)
            set_neurons.append(random.choice(neurons))
        model_layer_dict[tuple(set_neurons)] = False
    return True
    


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index

def set_to_cover(model_layer_dict):
    not_covered = [n_tuple for n_tuple, v in model_layer_dict.items() if not v]
    if not_covered:
        neurons_tuple = random.choice(not_covered)
    else:
        neurons_tuple = random.choice(model_layer_dict.keys())
    location_list = []
    for neuron in neurons_tuple:
        split_list = neuron.split('_')
        neuron_layer = ''
        for i in range(len(split_list)-1):
            if i == 0:
                neuron_layer = neuron_layer+split_list[i]
            else:
                neuron_layer = neuron_layer+'_'+split_list[i]
        neuron_location = [neuron_layer, int(split_list[len(split_list)-1])]
        location_list.append(neuron_location)
    return location_list

def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def set_covered(model_layer_dict):
    covered_set = len([v for v in model_layer_dict.values() if v])
    total_set = len(model_layer_dict)
    return covered_set, total_set, covered_set / float(total_set)


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True

def set_update_coverage(input_data, model, model_layer_dict, threshold=0):
    # prepare scaled output
    scaled_list = []
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        scaled_list.append(scaled)
    # update each set
    for neurons_tuple in model_layer_dict:
        active_result = True
        for neuron in neurons_tuple:
            split_list = neuron.split('_')
            neuron_layer = ''
            for i in range(len(split_list)-1):
                if i == 0:
                    neuron_layer = neuron_layer+split_list[i]
                else:
                    neuron_layer = neuron_layer+'_'+split_list[i]
            neuron_location = [neuron_layer, int(split_list[len(split_list)-1])]
            index_scaled_list = layer_names.index(neuron_location[0])
            if np.mean(scaled_list[index_scaled_list][..., neuron_location[1]]) <= threshold:
                active_result = False
        if (active_result == True) and (not model_layer_dict[neurons_tuple]):
            model_layer_dict[neurons_tuple] = True

def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False
