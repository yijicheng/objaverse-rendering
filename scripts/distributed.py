import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import os
import objaverse

import boto3
import tyro
import wandb


@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""

    # input_models_path: str
    # """Path to a json file containing a list of 3D object files"""
    start: int = 0
    end: int = 1
    lvis: bool = False
    cap3d_hq: bool = False
    output_dir: str = "./views"

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
    output_dir: str,
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        # Perform some operation on the item
        print(item, gpu)
        command = (
            # f"export DISPLAY=:0.{gpu} &&"
            f" CUDA_VISIBLE_DEVICES={gpu} "
            f" blender-3.2.2-linux-x64/blender -b -P scripts/blender_script.py --"
            f" --object_path {item} --output_dir {output_dir}"
        )
        print(command)
        subprocess.run(command, shell=True)

        if args.upload_to_s3:
            if item.startswith("http"):
                uid = item.split("/")[-1].split(".")[0]
                for f in glob.glob(f"views/{uid}/*"):
                    s3.upload_file(
                        f, "objaverse-images", f"{uid}/{f.split('/')[-1]}"
                    )
            # remove the views/uid directory
            shutil.rmtree(f"views/{uid}")

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering")

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, s3, args.output_dir)
            )
            process.daemon = True
            process.start()

    # Add items to the queue
    # with open(args.input_models_path, "r") as f:
    #     model_paths = json.load(f)

    if args.lvis:
        lvis_anno = objaverse.load_lvis_annotations()
        uids = []
        for cls, scenes in lvis_anno.items():
            uids.extend(scenes)
    elif args.cap3d_hq:
        import csv
        uids = []  
        with open('../datasets/hf-objaverse-v1/Cap3D_automated_Objaverse_highquality.csv', 'r') as csvfile:  
            csvreader = csv.reader(csvfile)  
            for row in csvreader:  
                uids.append(row[0])  
    else:
        uids = objaverse.load_uids()
    print("ALL: ", len(uids))
    uids = uids[args.start:args.end]
    print("SHARD: ", args.start, args.end)

    _ = objaverse.load_objects(
        uids=uids,
        download_processes=multiprocessing.cpu_count()
    )
    uid2paths = objaverse._load_object_paths()
    for uid in uids:
        queue.put(os.path.join(".objaverse/hf-objaverse-v1", uid2paths[uid]))

    # update the wandb count
    if args.log_to_wandb:
        while True:
            time.sleep(5)
            wandb.log(
                {
                    "count": count.value,
                    "total": len(uids),
                    "progress": count.value / len(uids),
                }
            )
            if count.value == len(uids):
                break

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)

    for uid in uids:
        os.system(f'rm -r {os.path.join(".objaverse/hf-objaverse-v1", uid2paths[uid])}')