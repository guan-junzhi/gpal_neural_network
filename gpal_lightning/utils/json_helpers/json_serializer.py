import json

import numpy as np


class JsonSerializer(json.JSONEncoder):
    """Helper class to help serialize dictionary"""

    def default(self, o):
        if isinstance(o, np.ndarray):
            o = o.squeeze()
            shape = o.shape
            o = np.array([float(obj) for obj in o.flatten()]).reshape(shape)
            o = o.tolist()
        elif isinstance(o, list):
            o = [float(obj) for obj in o]
        elif isinstance(o, (str, bool)):
            o = json.JSONEncoder.default(self, o)
        else:
            o = float(o)
        return o
