import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from catcvae.molgraph import atom_encoder_m, bond_encoder_m
import torch


# ref: https://gist.github.com/angeligareta/83d9024c5e72ac9ebc34c9f0b073c64c
def generateClassWeights(class_series, multi_class=True, one_hot_encoded=False):
  """
  Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
  Some examples of different formats of class_series and their outputs are:
    - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
    - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
    {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
    - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
    - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
    {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
  The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
  of appareance of the label when the dataset was processed. 
  In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
  Author: Angel Igareta (angel@igareta.com)
  """
  if multi_class:
    # If class is one hot encoded, transform to categorical labels to use compute_class_weight   
    if one_hot_encoded:
      class_series = np.argmax(class_series, axis=1)
  
    # Compute class weights with sklearn method
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    return dict(zip(class_labels, class_weights))
  else:
    # It is neccessary that the multi-label values are one-hot encoded
    mlb = None
    if not one_hot_encoded:
      mlb = MultiLabelBinarizer()
      class_series = mlb.fit_transform(class_series)

    n_samples = len(class_series)
    n_classes = len(class_series[0])

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1
    
    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))
  

def generateClassWeightsPosNeg(class_series):
  # compute number of positive divided by negative for all positions in class_series
  positive_count = np.sum(class_series == 1, axis=0)
  negative_count = np.sum(class_series == 0, axis=0)
  class_weights_pos_neg = negative_count / (positive_count + 0.0001)
  return np.array(class_weights_pos_neg)


def generateClassWeightsPosition(class_series):
  class_weights_pos_neg = list()
  for d in range(len(class_series[0])):
      class_series_d = list(class_series[:,d:d+1].reshape(len(class_series)))
      class_series_d.append(1.0) # prevent empty class
      class_series_d.append(0.0) # prevent empty class
      class_weights_pos_neg.append(compute_class_weight(class_weight="balanced", classes=[0,1], y=class_series_d)[1])
  return np.array(class_weights_pos_neg)
  

# Get class weights for length, annotation and adjacency
def getClassWeight(datasets_dobj_train, matrix_size, device='cpu'):
  len_atom_type = len(atom_encoder_m) 
  len_bond_type = len(bond_encoder_m)

  # length column
  cw_atom = list()
  for y in datasets_dobj_train:
      # print(y.matrix_catalyst[:, 0])
      cw_atom.append(y.matrix_catalyst[:, 0].tolist())
  class_weights_atom = generateClassWeightsPosition(np.array(cw_atom))
  class_weights_atom = torch.tensor(np.array(class_weights_atom), dtype=torch.float).to(device)
  # print(class_weights_atom)
  # print(class_weights_atom.shape)

  # annotation matrix
  cw_annotation = list()
  for y in datasets_dobj_train:
      # cw_annotation.extend(y.matrix_catalyst[:, 1:1+len_atom_type].reshape(-1, len_atom_type).tolist())
      mask_atom = np.argmax(y.matrix_catalyst[:, 0:1])
      anno = y.matrix_catalyst[:mask_atom, 1:1+len_atom_type].reshape(-1, len_atom_type).tolist()
      cw_annotation.extend(anno)
  class_weights_annotation = generateClassWeightsPosition(np.array(cw_annotation))
  class_weights_annotation = torch.tensor(np.array(class_weights_annotation), dtype=torch.float).to(device)
  # print(class_weights_annotation)
  # print(class_weights_annotation.shape)

  # adjacency matrix
  cw_adjacency = list()
  for y in datasets_dobj_train:
      # cw_adjacency.extend(y.matrix_catalyst[:, len_atom_type+1:].reshape(-1, len_bond_type).tolist())
      mask_atom = np.argmax(y.matrix_catalyst[:, 0:1])
      adj = y.matrix_catalyst[:mask_atom, len_atom_type+1:len_atom_type+1+(mask_atom*len_bond_type)].reshape(-1, len_bond_type).tolist()
      cw_adjacency.extend(adj)
  class_weights_adjacency = generateClassWeightsPosition(np.array(cw_adjacency))
  class_weights_adjacency = torch.tensor(np.array(class_weights_adjacency), dtype=torch.float).to(device)
  # print(class_weights_adjacency)
  # print(class_weights_adjacency.shape)

  class_weights = {"atom": class_weights_atom, "annotation": class_weights_annotation, "adjacency": class_weights_adjacency}
  return class_weights
   