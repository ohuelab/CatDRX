dataset_args = dict()

dataset_args['ord'] = {
    'file': 'ord',
    'smiles': {'reactant': 'Reactant_SMILES_new', 
               'reagent': 'Reagent_SMILES_new',
               'product': 'Product_SMILES_new', 
               'catalyst': 'Catalyst_SMILES_new'},
    'task': 'Yield_Clipped',
    'ids': 'index',
    'splitting': None,
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'time': 'Time_h',
    'condition_dict': {'Catalyst_SMILES_mw':  {'type': 'continuous', 
                                               'list': [100, 200, 300, 400, 500, 1000]}},
}

dataset_args['bh_test1'] = {
    'file': 'bh_test1',
    'smiles': {'reactant': 'rxn_reactants', 
               'reagent': 'rxn_reagents', 
               'product': 'rxn_product', 
               'catalyst': 'rxn_catalyst'},
    'task': 'Output',
    'ids': 'index',
    'splitting': 'Test1',
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'time': 'Time_h',
    'condition_dict': {'mw_catligand':  {'type': 'continuous', 
                                       'list': [100, 200, 300, 400, 500, 1000]}},
}

dataset_args['sm_test1'] = {
    'file': 'sm_test1',
    'smiles': {'reactant': 'reactant', 
               'reagent': 'reagent', 
               'product': 'product', 
               'catalyst': 'catalyst'},
    'task': 'y',
    'ids': 'index',
    'splitting': 'Test1',
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'time': 'Time_h',
    'condition_dict': {'mw_catligand':  {'type': 'continuous', 
                                       'list': [100, 200, 300, 400, 500, 1000]}},
}

dataset_args['l_sm'] = {
    'file': 'l_sm',
    'smiles': {'reactant': 'Input1_Input2', 
               'reagent': 'PdCatalyst_Additive',
               'product': 'Product', 
               'catalyst': 'Ligand'},
    'task': 'Output',
    'ids': 'index',
    'splitting': None,
    'predictiontask': 'yield',
    'predictiontype': 'regression',
    'time': 'Time_h',
    'condition_dict': {'mw_ligand':  {'type': 'continuous', 
                                       'list': [100, 200, 300, 400, 500, 1000]}},
}

dataset_args['ps'] = {
    'file': 'ps',
    'smiles': {'reactant': 'Sub_SMILES_new', 
               'reagent': 'Reagent_SMILES_new',
               'product': 'Product_SMILES_new', 
               'catalyst': 'Cat_SMILES_new'},
    'task': 'DeltaDeltaG',
    'ids': 'index',
    'splitting': None,
    'predictiontask': 'others',
    'predictiontype': 'regression',
    'time': 'time',
    'condition_dict': {'mw_cat':  {'type': 'continuous', 
                                   'list': [100, 200, 300, 400, 500, 1000]}}
}




