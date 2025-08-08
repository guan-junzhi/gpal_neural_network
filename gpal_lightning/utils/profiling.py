import time
import numpy as np
import logging
import os
import psutil


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


def GetMemInfo():
    a = os.popen("free -m").readlines()
    titles = [ele for ele in a[0].replace('\n', '').split(' ') if ele != '']
    values = [ele for ele in a[1].replace(
        '\n', '').split(' ') if ele != ''][1:]

    return {t: float(v) / 1024.0 for t, v, in zip(titles, values)}


def PrintTopProcesses(num_processes=10):
    processes = psutil.process_iter(
        ['pid', 'name', 'cpu_percent', 'memory_percent'])
    processes = sorted(processes, key=lambda x: x.info['cpu_percent'], reverse=True)[
        :num_processes]
    logging.warning(f"{'PID':>8}  {'Name':<20}  {'CPU%':>10}  {'MEM%':>10}")
    for process in processes:
        logging.warning(
            f"{process.info['pid']:>8}  {process.info['name']:<20}  {process.info['cpu_percent']:>10.2f}  {process.info['memory_percent']:>10.2f}")


class TrainSpeedRec():
    def __init__(self, qlen=1000):
        self.qlen = qlen
        self.rec_q = []

    def Rec(self, iter):
        self.rec_q.append([iter, time.time()])
        self.rec_q = self.rec_q[-self.qlen:]

    def GetAvg(self, window_width=100):
        if len(self.rec_q) < 2:
            return None
        last = self.rec_q[-1]
        for t in self.rec_q[::-1][1:]:
            if (last[0] - t[0]) >= window_width:
                di = last[0] - t[0]
                dts = last[1] - t[1]
                iter_per_hour = di / dts * 3600
                return iter_per_hour
        return None
