import time
import numpy as np
import logging


def TimeProf(func):
    def Warp(*args, **kwargs):
        t1 = time.time()
        func_ret = func(*args, **kwargs)
        t2 = time.time()
        if isinstance(func_ret, dict):
            func_ret.update({"dataloader_time": np.array([t2-t1])})
        return func_ret

    return Warp


class DetailProf:
    def __init__(self, init_key="prof_instance_init"):
        self.tics = {init_key: time.time()}
        self.durs = {}

    def AddSubProf(self, key, prof):
        self.durs[key] = prof

    def Tic(self, key):
        self.tics[key] = time.time()

    def Duration(self, key, from_key=None):
        if from_key == None:
            from_key = list(self.tics.keys())[-1]
        if from_key not in self.tics:
            return
        if key not in self.tics:
            self.tics[key] = time.time()

        self.durs[key] = self.tics[key] - self.tics[from_key]
        return

    def PrintOne(self, key, t, level=0):
        logging.warning(f"{'-----' * level}{key}: {t: < .4}")

    def Print(self, comment=None, level=0):
        sum_dura = 0
        for k in self.durs:
            if isinstance(self.durs[k], type(self)):
                sum_sub = self.durs[k].Print(None, level + 1)
                if k+"_sum" in self.durs:
                    sum_sub = self.durs[k+"_sum"]
                sum_dura += sum_sub
                self.PrintOne(k+"_sum", sum_sub, level)
            else:
                self.PrintOne(k, self.durs[k], level)
                sum_dura += self.durs[k]
        if comment is not None:
            self.PrintOne(f"***{comment}***", sum_dura, 0)

        return sum_dura

    def NowFrom(self, from_key):
        return time.time() - self.tics[from_key]
