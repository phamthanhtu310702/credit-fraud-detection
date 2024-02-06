import lmdb
import numpy as np
import pickle
class FeatureStore(object):
    def __init__(self, path, map_size = int(10e7)):
        self.db = lmdb.open(path, map_size=map_size)
    
    def _key(self, key):
        return str(key).encode('utf-8')

    def put(self, key, value, dtype=np.float32, wb=None):
        value = value.astype(dtype)
        obj = wb
        obj.put(self._key(key), value.tobytes())

    def get(self, key, default_value, dtype=np.float32, wb=None):
        obj = wb
        rval = obj.get(self._key(key))
        if rval is None:
            return default_value
        return np.frombuffer(rval, dtype=dtype)