import json
import os

import numpy as np

GOLDENS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")


def load_case(name):
    npz = np.load(os.path.join(GOLDENS_DIR, name + ".npz"))
    assert int(npz["schema_version"]) == 1, "golden schema mismatch — regenerate goldens"
    meta = json.loads(bytes(npz["meta_json"].tobytes()).decode("utf-8"))
    return npz, meta


def state_dict_of(npz):
    return {k[len("sd/"):]: npz[k] for k in npz.files if k.startswith("sd/")}


def grads_of(npz):
    return {k[len("grad/"):]: npz[k] for k in npz.files if k.startswith("grad/")}
