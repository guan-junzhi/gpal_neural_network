import os
import pickle
import time
import mmap
import logging
import lmdb


class FastLoaderBuffer():
    def __init__(self, filename) -> None:
        self.lmdb_path = filename
        self.map_size = 1099511627776 * 4
        self.lmdb = None

    def Cache(self, key, data):
        try:
            # if True:
            with lmdb.open(self.lmdb_path, map_size=self.map_size, lock=True) as local_access:
                with local_access.begin(write=True) as txn:
                    ret = txn.put(str(key).encode(), data)
                    # print(f"cache {key} success")
                    return True
        except Exception as e:
            print(f"cache {key} faild {e}")
        return False

    def __getitem__(self, key):
        with lmdb.open(self.lmdb_path, map_size=self.map_size, lock=False) as local_access:
            with local_access.begin(write=False) as txn:
                x = txn.get(str(key).encode())
                if x is None:
                    raise ValueError(
                        "invalid access ValueError = ", key, "in ", self.lmdb_path)
                data = x

        # print("valid access = ", key, "in ", self.lmdb_path)
        return data
