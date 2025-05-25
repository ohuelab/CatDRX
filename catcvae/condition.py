import numpy as np

# condition

def getOneHotDiscrete(value, range_list):
    assert value in range_list, "Value not in the range"
    return [1 if value == r else 0 for r in range_list]

def getOneHotContinuous(value, range_list):
    # assert value >= range[0] and value <= range[-1], "Value not in the range"
    onehot = [0 for _ in range(len(range_list)+1)]
    for i in range_list:
        if value < i:
            onehot[range_list.index(i)] = 1
            break
    if onehot == [0 for _ in range(len(range_list)+1)]:
        onehot[-1] = 1
    return onehot

def getOneHotCondition(condition, condition_dict):
    onehot = []
    for key in condition_dict.keys():
        if condition_dict[key]['type'] == 'discrete':
            onehot = onehot + getOneHotDiscrete(condition[key], condition_dict[key]['list'])
        elif condition_dict[key]['type'] == 'continuous':
            onehot = onehot + getOneHotContinuous(condition[key], condition_dict[key]['list'])
        else:
            raise ValueError("Invalid condition type")
    return onehot

def getConditionDim(condition_dict):
    dim = 0
    for key in condition_dict.keys():
        if condition_dict[key]['type'] == 'discrete':
            dim += len(condition_dict[key]['list'])
        elif condition_dict[key]['type'] == 'continuous':
            dim += len(condition_dict[key]['list'])+1
        else:
            raise ValueError("Invalid condition type")
    return dim

def getSampleCondition(condition, condition_dict):
    onehot = []
    for key in condition_dict.keys():
        if condition_dict[key]['type'] == 'discrete':
            if condition[key] is None:
                # random onehot from len
                sampling = [0 for _ in condition_dict[key]['list']]
                sampling[np.random.randint(0, len(condition_dict[key]['list']))] = 1
                onehot = onehot + sampling
            else:
                onehot = onehot + getOneHotDiscrete(condition[key], condition_dict[key]['list'])
        elif condition_dict[key]['type'] == 'continuous':
            if condition[key] is None:
                # random onehot from len
                sampling = [0 for _ in range(len(condition_dict[key]['list'])+1)]
                sampling[np.random.randint(0, len(condition_dict[key]['list'])+1)] = 1
                onehot = onehot + sampling
            else:
                onehot = onehot + getOneHotContinuous(condition[key], condition_dict[key]['list'])
        else:
            raise ValueError("Invalid condition type")
    return onehot
