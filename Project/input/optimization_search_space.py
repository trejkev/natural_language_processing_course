from hyperopt import hp

cList = []
for number in range(1, 1000):
    cList.append(number / 10)
gammaList = cList[:]
gammaList.append("scale")

# -- Search space for Support Vector Machine classifier
svc_search_space = {
    "kernel"                 : hp.choice(
        "kernel", ['linear', 'poly', 'rbf', 'sigmoid']
    ),
    "C"                      : hp.choice("C", cList),
    "gamma"                  : hp.choice("gamma", gammaList),
    "decision_function_shape": hp.choice(
        "decision_function_shape", ['ovo', 'ovr']
    ),
    "degree"                 : hp.choice("degree", list(range(2, 10)))
}

# -- Search space for Neural Network with Word Embeddings
nn_search_space = {
    "vector_size"        : hp.choice("vector_size", [50, 100, 200]),
    "window_size"        : hp.choice("window_size", list(range(5, 21))),
    "sg"                 : hp.choice("sg", [0, 1]),
    "hidden_layers"      : hp.choice("hidden_layers", list(range(1, 7))),
    "neurons"            : hp.choice("neurons", [64, 128, 256]),
    "activation_function": hp.choice(
        "activation_function", ["relu", "tanh", "sigmoid"]
    ),
    "batch_size"         : hp.choice("batch_size", list(range(32, 129))),
    "optimizer"          : hp.choice("optimizer", ["adam", "sgd", "rmsprop"]),
    "epochs"             : hp.choice("epochs", list(range(10, 200))) 
}

# -- Default params for Support Vector Machine classifier
svc_default_params = {
    "kernel"                 : 'rbf',
    "C"                      : 10,
    "gamma"                  : 0.1,
    "decision_function_shape": 'ovo',
    "degree"                 : 2
}

# -- Default params for Neural Network with Word Embeddings
nn_default_params = {
    "vector_size"        : 200,
    "window_size"        : 5,
    "sg"                 : 1,
    "hidden_layers"      : 4,
    "neurons"            : 128,
    "activation_function": 'relu',
    "batch_size"         : 32,
    "optimizer"          : 'sgd',
    "epochs"             : 100 
}