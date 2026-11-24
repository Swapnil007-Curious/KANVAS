"""Write dataset-metadata.json for the Kaggle dataset, with README.md as its description.

Run from the repository root after editing README.md:

    python kaggle/datasets/alibaba-gpu-trace-2020/make_metadata.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def field(name, desc, typ="string"):
    return {"name": name, "description": desc, "type": typ}


F = field
RESOURCES = [
    {"path": "pai_job_table.csv", "description": "One row per job: user, status, submission and completion time.",
     "schema": {"fields": [F("job_name", "job id (hash)"), F("inst_id", "job instance id; one per job (hash)"),
                           F("user", "submitting user (hash)"), F("status", "Terminated, Failed, Running or Waiting"),
                           F("start_time", "submission time, trace seconds", "number"),
                           F("end_time", "completion time, trace seconds", "number")]}},
    {"path": "pai_task_table.csv", "description": "One row per task of a job: status, times and resource requests.",
     "schema": {"fields": [F("job_name", "job id (hash)"), F("task_name", "task role, e.g. tensorflow, worker, ps"),
                           F("inst_num", "number of instances requested", "number"),
                           F("status", "Terminated, Failed, Running or Waiting"),
                           F("start_time", "task launch time (equals the job's submission time), trace seconds", "number"),
                           F("end_time", "task end time, trace seconds", "number"),
                           F("plan_cpu", "requested CPU in percent of a core (600 = 6 cores)", "number"),
                           F("plan_mem", "requested memory, GB", "number"),
                           F("plan_gpu", "requested GPU per instance in percent of one GPU (50 = half a GPU)", "number"),
                           F("gpu_type", "GPU model of the machine the task ran on; MISC = older GPUs")]}},
    {"path": "pai_instance_table.csv", "description": "One row per instance (running copy of a task): status, times, machine.",
     "schema": {"fields": [F("job_name", "job id (hash)"), F("task_name", "task role"), F("inst_name", "instance name (hash)"),
                           F("worker_name", "worker id; joins to sensor and machine_metric (hash)"),
                           F("inst_id", "job instance id (hash)"), F("status", "instance status"),
                           F("start_time", "instance launch time, trace seconds", "number"),
                           F("end_time", "instance end time, trace seconds", "number"), F("machine", "machine id (hash)")]}},
    {"path": "pai_sensor_table.csv",
     "description": "Per-instance resource use averaged over the instance's life (max_* columns are maxima).",
     "schema": {"fields": [F("job_name", "job id (hash)"), F("task_name", "task role"), F("worker_name", "worker id (hash)"),
                           F("inst_id", "job instance id (hash)"), F("machine", "machine id (hash)"),
                           F("gpu_name", "GPU device on the machine, e.g. /dev/nvidia0"),
                           F("cpu_usage", "CPU used in percent of a core", "number"),
                           F("gpu_wrk_util", "GPU used in percent of one GPU", "number"),
                           F("avg_mem", "mean main memory, GB", "number"), F("max_mem", "peak main memory, GB", "number"),
                           F("avg_gpu_wrk_mem", "mean GPU memory, GB", "number"),
                           F("max_gpu_wrk_mem", "peak GPU memory, GB", "number"),
                           F("read", "network input, bytes", "number"), F("write", "network output, bytes", "number"),
                           F("read_count", "number of reads", "number"), F("write_count", "number of writes", "number")]}},
    {"path": "pai_group_tag_table.csv", "description": "Per-job tags: requested GPU type spec, repeated-job group, workload name.",
     "schema": {"fields": [F("inst_id", "job instance id (hash)"), F("user", "user (hash)"),
                           F("gpu_type_spec", "GPU type the job asked for; empty if none"),
                           F("group", "semantic tag; jobs with the same tag are repeats (hash)"),
                           F("workload", "workload name, e.g. bert, ctr; set for about 10% of jobs")]}},
    {"path": "pai_machine_spec.csv", "description": "One row per machine: GPU type and capacity.",
     "schema": {"fields": [F("machine", "machine id (hash)"), F("gpu_type", "GPU model, or CPU for CPU-only machines"),
                           F("cap_cpu", "CPU cores", "number"), F("cap_mem", "memory, GB", "number"),
                           F("cap_gpu", "number of GPUs", "number")]}},
    {"path": "pai_machine_metric.csv", "description": "Machine-level telemetry averaged over one instance's life on that machine.",
     "schema": {"fields": [F("worker_name", "worker id (hash)"), F("machine", "machine id (hash)"),
                           F("start_time", "instance launch time, trace seconds", "number"),
                           F("end_time", "instance end time, trace seconds", "number"),
                           F("machine_cpu_iowait", "CPU iowait", "number"), F("machine_cpu_kernel", "CPU kernel time", "number"),
                           F("machine_cpu_usr", "CPU user time", "number"),
                           F("machine_gpu", "machine GPU utilisation summed over GPUs (100 = one busy GPU)", "number"),
                           F("machine_load_1", "1-minute load average", "number"),
                           F("machine_net_receive", "network bytes received", "number"),
                           F("machine_num_worker", "workers on the machine", "number"),
                           F("machine_cpu", "CPU utilisation", "number")]}},
    {"path": "kanvas_tasks_1h.parquet", "description": "Derived: 539,816 labelled GPU tasks with the ten KANVAS features (see README).",
     "schema": {"fields": [F("job_name", "job id (hash)"), F("task_name", "task role"),
                           F("plan_cpu", "requested CPU", "number"), F("plan_mem", "requested memory, GB", "number"),
                           F("plan_gpu", "requested GPU per instance", "number"), F("inst_num", "instances requested", "number"),
                           F("avg_cpu_usr", "machine CPU user time, mean over other jobs in the previous hour", "number"),
                           F("avg_gpu_util", "machine GPU utilisation, same window", "number"),
                           F("avg_cpu_kernel", "machine CPU kernel time, same window", "number"),
                           F("avg_load_1", "machine load, same window", "number"),
                           F("cap_gpu", "GPUs on the machine", "number"),
                           F("avg_net_receive", "machine network bytes received, same window", "number"),
                           F("label", "1 = Failed, 0 = Terminated", "integer"),
                           F("start_time", "first instance launch time, trace seconds", "number"),
                           F("user_rank", "user as an integer, 1 = most tasks", "integer"),
                           F("machine", "machine id (hash)"), F("n_metric_rows", "telemetry rows averaged", "integer")]}},
    {"path": "LICENSE.txt", "description": "Creative Commons Attribution 4.0 International licence text, from the trace's LICENSE file."},
    {"path": "README.md", "description": "Source, attribution, citation, changes from the official release and column notes."},
    {"path": "SHA256SUMS.txt", "description": "Checksums of each CSV and of the official headerless file it contains."},
]


def main():
    meta = {
        "title": "Alibaba PAI GPU cluster trace 2020",
        "id": "swapnilshukla007/alibaba-gpu-trace-2020",
        "subtitle": "Jobs, tasks, instances and telemetry of a 6,742-GPU ML cluster (CC BY 4.0)",
        "description": (HERE / "README.md").read_text(encoding="utf-8"),
        "isPrivate": True,
        "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["computer science", "tabular", "time series analysis", "data analytics"],
        "collaborators": [],
        "data": [],
        "resources": RESOURCES,
    }
    (HERE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote dataset-metadata.json")


if __name__ == "__main__":
    main()
