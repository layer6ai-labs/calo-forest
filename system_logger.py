import psutil
import time
import os

import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Process


TAG = "system_info"

def log(logdir, delay):
    # initial information
    cpu_count = psutil.cpu_count()
    total_mem = psutil.virtual_memory().total>>20
    t1 = time.time()
    lines = [f"cpu_count: {cpu_count} total_memory (MiB): {total_mem}\n",
             "time,cpu_percent,mem_percent,available,used,cache,shared,buffers\n"]
    path = os.path.join(logdir, f"{TAG}.txt")
    with open(path, "w", buffering=1) as f:
        f.write(lines[0])
        f.write(lines[1])

        while True:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            t2 = time.time()
            line = f"{t2-t1:.1f},{cpu},{mem.percent},{mem.available>>20},{mem.used>>20},{mem.cached>>20},{mem.shared>>20},{mem.buffers>>20}\n"
            f.write(line)
            time.sleep(delay)

class SystemLogger():

    def __init__(self, logdir, delay):
        self.logdir = logdir
        self.delay = delay

    def start(self):
        self.p = Process(target=log, args=(self.logdir, self.delay))
        self.p.start()

    def on(self):
        return True if self.p else False

    def finish(self):
        if self.p:
            self.p.kill()
            time.sleep(0.5)
            self.p.close()
            self.p = None

    def plot_system_usage(self):
        # Clean up any other plots that were created
        plt.close('all')
        plt.cla()
        plt.clf()

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16,}

        def smooth(scalars, weight): # Weight between 0 and 1
            last = scalars[0]
            smoothed = list()
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val

            return smoothed

        read_path = os.path.join(self.logdir, f"{TAG}.txt")
        with open(read_path, "r") as f:
            line = f.readline()
        lst = line.split()
        cpu_count = int(lst[1])
        total_mem = int(lst[4])
        df = pd.read_csv(read_path, header=1)

        df['Percent'] = df['mem_percent']
        df['Available'] = (df['available'] / total_mem) * 100
        df['tot_mem'] = (total_mem - df['available']) / 1024.0
        max_mem = round(df['tot_mem'].max(), 1)
        min_mem = round(df['tot_mem'].min(), 1)

        df['Used'] = (df['used'] / total_mem) * 100
        df['Cache'] = (df['cache'] / total_mem) * 100
        df['Shared'] = (df['shared'] / total_mem) * 100
        df['Buffers'] = (df['buffers'] / total_mem) * 100
        df['tot_mem_custom'] = (df['used'] + df['cache'] + df['buffers']) / 1024.0
        
        max_mem_old = round(df['tot_mem_custom'].max(), 1)
        min_mem_old = round(df['tot_mem_custom'].min(), 1)
        mins = False
        final_time = df['time'].iloc[-1]
        if final_time > 3600:
            df['time'] = df['time'] // 60
            mins = True
            final_time = df['time'].iloc[-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.stackplot(df['time'], df['Percent'],
                    colors=['#3b528b'], # '#2596be'
                    labels=['Memory'])
        plt.plot(df['time'], smooth(df['cpu_percent'], 0.3), color='black', label='CPU', linewidth=2)
        plt.legend(loc='lower left', fontsize=14)
        xlabel = 'Time (m)' if mins else 'Time (s)'
        plt.xlabel(xlabel, fontdict=font)
        plt.ylabel('Resource Usage (%)', fontdict=font)
        gbs = round(total_mem / 1024, 1)
        plt.title(f'Resource Usage over Time ({cpu_count} CPUs, {gbs}GiB RAM)', fontdict=font)
        plt.xlim([0, final_time])
        plt.ylim([0, 100.2])
        plt.xticks(fontfamily="serif", fontsize=14)
        plt.yticks(fontfamily="serif", fontsize=14)
        text = f"Min Mem: {min_mem} GiB\nMax Mem: {max_mem} GiB"
        plt.text(0.885, 0.075, text, bbox=dict(facecolor='white', alpha=0.7), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, fontdict=font)
        save_path = os.path.join(self.logdir, 'system_usage.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches=None, pad_inches=0.0)

        plt.close('all')
        plt.cla()
        plt.clf()

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.stackplot(df['time'], df['Used'], df['Cache'], df['Buffers'],
                    colors=['#3b528b', '#5ec962', '#21918C'], # '#2596be'
                    labels=['Used', 'Cache', 'Buffers'])
        plt.plot(df['time'], smooth(df['cpu_percent'], 0.3), color='black', label='CPU', linewidth=2)
        plt.legend(loc='lower left', fontsize=14)
        xlabel = 'Time (m)' if mins else 'Time (s)'
        plt.xlabel(xlabel, fontdict=font)
        plt.ylabel('Resource Usage (%)', fontdict=font)
        gbs = round(total_mem / 1024, 1)
        plt.title(f'Resource Usage over Time ({cpu_count} CPUs, {gbs}GiB RAM)', fontdict=font)
        plt.xlim([0, final_time])
        plt.ylim([0, 100.2])
        plt.xticks(fontfamily="serif", fontsize=14)
        plt.yticks(fontfamily="serif", fontsize=14)
        text = f"Min Mem: {min_mem_old} GiB\nMax Mem: {max_mem_old} GiB"
        plt.text(0.885, 0.075, text, bbox=dict(facecolor='white', alpha=0.7), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, fontdict=font)
        save_path = os.path.join(self.logdir, 'system_usage_old.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches=None, pad_inches=0.0)

        return max_mem, min_mem
