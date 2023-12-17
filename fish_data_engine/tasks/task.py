from abc import abstractmethod
from copy import deepcopy
import pickle
import time
from loguru import logger
import os
import sys
from tempfile import TemporaryDirectory
import torch
import subprocess as sp
from collections import Counter

RANK = os.environ.get("RANK", None)
JOB_PICKLE = os.environ.get("JOB_PICKLE", None)
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_WORKER = RANK is not None
CURR_WORKER = f"[Rank {RANK}]" if IS_WORKER else "[Main]"


class Task:
    USE_CUDA = True

    @abstractmethod
    def jobs(self) -> list:
        # This will be executed in the main process
        raise NotImplementedError

    @abstractmethod
    def process(self, job):
        # This will be executed in the worker processes
        raise NotImplementedError

    def fire_child(self, rank, job_pickle):
        env = deepcopy(os.environ)

        # Respect CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES" in env:
            devices = env["CUDA_VISIBLE_DEVICES"].split(",")
            env["CUDA_VISIBLE_DEVICES"] = devices[rank % len(devices)]
        elif self.USE_CUDA:
            env["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

        # Launch child process
        env["RANK"] = str(rank)
        env["JOB_PICKLE"] = job_pickle
        env["WORLD_SIZE"] = str(WORLD_SIZE)
        return sp.Popen([sys.executable] + sys.argv, env=env)

    def launch_child_processes(self):
        # Main process
        logger.info(f"{CURR_WORKER} Running in main process")
        jobs = self.jobs()
        logger.info(f"{CURR_WORKER} {len(jobs)} jobs to process")
        processes = []

        with TemporaryDirectory() as tmpdir:
            for rank in range(WORLD_SIZE):
                subset = jobs[rank::WORLD_SIZE]

                # Dump jobs to pickle file
                job_pickle = os.path.join(tmpdir, f"jobs_{rank}.pkl")
                logger.info(f"{CURR_WORKER} Assign {len(subset)} jobs to {job_pickle}")
                with open(job_pickle, "wb") as f:
                    pickle.dump(subset, f)

                # Launch child process
                p = self.fire_child(rank, job_pickle)
                processes.append(p)
                logger.info(f"{CURR_WORKER} Launched child process {rank}")

            # Wait for all child processes to finish
            while any([p is not True for p in processes]):
                for idx, p in enumerate(processes):
                    if p.poll() is None:
                        continue

                    if p.returncode != 0:
                        logger.error(
                            f"{CURR_WORKER} Worker {idx} failed with code {p.returncode}, relaunching"
                        )
                        p = self.fire_child(idx, job_pickle)
                        processes[idx] = p
                        logger.info(f"{CURR_WORKER} Launched child process {idx}")
                    else:
                        logger.info(f"{CURR_WORKER} Worker {idx} finished successfully")
                        processes[idx] = True

        logger.info(f"{CURR_WORKER} All child processes finished successfully")

    def process_jobs(self):
        logger.info(f"{CURR_WORKER} Running in worker process")
        with open(JOB_PICKLE, "rb") as f:
            jobs = pickle.load(f)
        logger.info(f"{CURR_WORKER} {len(jobs)} jobs to process")

        log_time = 0
        start_time = time.time()
        counter = Counter()

        for idx, job in enumerate(jobs):
            status = self.process(job)
            if status is None:
                status = "success"

            counter[status] += 1

            if time.time() - log_time > 10:
                eta_seconds = (
                    (time.time() - start_time) / (idx + 1) * (len(jobs) - idx - 1)
                )
                # Convert seconds to days, hours, minutes
                eta_days = eta_seconds // (24 * 3600)
                eta_hours = (eta_seconds % (24 * 3600)) // 3600
                eta_minutes = (eta_seconds % 3600) // 60
                eta_str = f"{eta_days:.0f}d {eta_hours:.0f}h {eta_minutes:.0f}m"

                status_str = ", ".join([f"{k}: {v}" for k, v in counter.items()])
                logger.info(
                    f"{CURR_WORKER} Processed {jobs.index(job)}/{len(jobs)} jobs, ETA: {eta_str} -> {status_str}"
                )
                log_time = time.time()

        logger.info(f"{CURR_WORKER} All jobs finished successfully")

    def run(self):
        if RANK is None:
            self.launch_child_processes()
        else:
            self.process_jobs()


if __name__ == "__main__":

    class DummyTask(Task):
        def jobs(self):
            return list(range(100))

        def process(self, job):
            time.sleep(0.1)
            return "success"

    DummyTask().run()
