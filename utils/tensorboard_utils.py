import os
import torch
from queue import Queue
from threading import Thread 
import subprocess
import shutil


class GPUStatsMonitor():
    """modified from pytorch_lightning/callbacks/gpu_stats_monitor"""
    def __init__(self, 
                 fan_speed: bool = False,
                 temperature: bool = False,):
        self.fan_speed = fan_speed
        self.temperature = temperature
        self.gpu_ids = self._get_gpu_ids()
        self.stat_keys = self._get_gpu_stat_keys()
        self.queries = [k for k, _ in self.stat_keys]

    def _get_gpu_stat_keys(self):
        """Get the GPU stats keys."""
        stat_keys = []

        stat_keys.append(("utilization.gpu", "%"))
        stat_keys.extend([("memory.used", "MB"), ("memory.free", "MB"), ("utilization.memory", "%")])
        if self.fan_speed:
            stat_keys.append(("fan.speed", "%"))
        if self.temperature:
            stat_keys.extend([("temperature.gpu", "°C"), ("temperature.memory", "°C")])
        return stat_keys

    @staticmethod
    def _get_gpu_ids():
        """Get the unmasked real GPU IDs."""
        # All devices if `CUDA_VISIBLE_DEVICES` unset
        default = ",".join(str(i) for i in range(torch.cuda.device_count()))
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
        return cuda_visible_devices

    def _get_gpu_stats(self):
        """Run nvidia-smi to get the gpu stats"""
        gpu_query = ",".join(self.queries)
        format = "csv,nounits,noheader"
        gpu_ids = ",".join(self.gpu_ids)
        result = subprocess.run(
            [
                # it's ok to supress the warning here since we ensure nvidia-smi exists during init
                shutil.which("nvidia-smi"),  # type: ignore
                f"--query-gpu={gpu_query}",
                f"--format={format}",
                f"--id={gpu_ids}",
            ],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True,
        )

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except ValueError:
                return 0.0
        stats = [[_to_float(x) for x in s.split(", ")] for s in result.stdout.strip().split(os.linesep)]
        return stats

    def get_stat(self):
        """get gpu stats, return as dictionary for logging"""
        stats = self._get_gpu_stats()
        output = {}
        for idx, item in enumerate(stats):
            output[idx] = {q: v for q, v in zip(self.queries, item)}
        return output


class PlotterThread():
    def __init__(self, writer):
        self.writer = writer
        self.gpu_monitor = GPUStatsMonitor()
        self.task_queue = Queue(maxsize=128)
        worker = Thread(target=self.do_work, args=(self.task_queue,))
        worker.setDaemon(True)
        worker.start()

    def do_work(self, q):
        while True:
            content = q.get()
            if content[-1] == 'image':
                self.writer.add_image(*content[:-1])
            elif content[-1] == 'scalar':
                self.writer.add_scalar(*content[:-1])
            elif content[-1] == 'scalars':
                self.writer.add_scalars(*content[:-1])
            elif content[-1] == 'figure':
                self.writer.add_figure(*content[:-1], close=True)
            else:
                raise ValueError
            q.task_done()

    def add_data(self, name, value, step, data_type='scalar'):
        self.task_queue.put([name, value, step, data_type])

    def log_gpustat(self, step):
        stats = self.gpu_monitor.get_stat()
        for device_id, stat_dict in stats.items():
            for k, v in stat_dict.items():
                self.add_data(f'device/{device_id}/{k}', v, step, data_type='scalar')

    def __len__(self):
        return self.task_queue.qsize()


class PlotterDummy():
    """plotter without threads"""
    def __init__(self, writer):
        self.writer = writer

    def do_work(self, content):
        if content[-1] == 'image':
            self.writer.add_image(*content[:-1])
        elif content[-1] == 'scalar':
            self.writer.add_scalar(*content[:-1])
        elif content[-1] == 'scalars':
            self.writer.add_scalars(*content[:-1])
        elif content[-1] == 'figure':
            self.writer.add_figure(*content[:-1], close=True)

    def add_data(self, name, value, step, data_type='scalar'):
        self.do_work([name, value, step, data_type])

    def log_gpustat(self, step):
        return


if __name__ == '__main__':
    monitor = GPUStatsMonitor(True, True)
    print(monitor.get_stat())