#!/usr/bin/env python3
"""Submit one Snakemake jobscript to HTCondor and print its cluster id.

Snakemake (7.x, legacy --cluster interface) invokes this as

    cluster/condor_submit.py "<jobscript>"

and reads the FIRST line of stdout as the external job id, so nothing else may be
written there. Diagnostics go to stderr.

The job's properties are embedded in the jobscript as a '# properties = {...}'
JSON comment. A group job carries "type": "group" and a "groupid", and has no
"rule" or "wildcards" key, so neither of those may be read unconditionally.
Note also that Snakemake omits the "_cores" and "_nodes" resources from that
dictionary: the core count comes from the top-level "threads" key instead.
"""

# imports
import os
import re
import subprocess
import sys

from snakemake.utils import read_job_properties

# Fallbacks used only when a rule declares neither the resource itself nor a
# '--default-resources' value for it. The profile sets defaults for all three, so
# reaching these means a rule was submitted without going through the profile.
DEFAULT_MEM_MB = 4000
DEFAULT_DISK_MB = 10000
DEFAULT_RUNTIME_MIN = 180

# Machine requirements.
# 'should_transfer_files = NO' below makes condor_submit add the matching
# FileSystemDomain requirement itself, which is what keeps jobs on the nodes that
# mount /vols and off the hep-hx3-batch-* opportunistic nodes, each of which sits
# in a filesystem domain of its own.
# 'has_avx' keeps the AVX-compiled xgboost and numpy wheels off the pool's older
# nodes, where they abort with "Illegal instruction". Note that this narrows the
# set of matching machines and relaxes nothing.
REQUIREMENTS = "has_avx"


def job_name(properties):
    """A filesystem- and ClassAd-safe name for the job, used for logs and condor_q."""
    if properties.get("type") == "group":
        # Group jobs have no rule name; the group id is the only handle they carry.
        name = "group_{}".format(properties.get("groupid", "unknown"))
    else:
        name = properties.get("rule", "snakejob")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def as_int(value, default):
    """A resource as a positive integer. Values may arrive as int, float or str."""
    try:
        return max(int(float(value)), 1)
    except (TypeError, ValueError):
        return default


def submit_description(jobscript, properties):
    """Return the HTCondor submit description for one Snakemake jobscript."""
    name = job_name(properties)
    resources = properties.get("resources", {}) or {}

    threads = as_int(properties.get("threads"), 1)
    mem_mb = as_int(resources.get("mem_mb"), DEFAULT_MEM_MB)
    disk_mb = as_int(resources.get("disk_mb"), DEFAULT_DISK_MB)
    # Snakemake's 'runtime' resource is in minutes, whereas '+MaxRuntime' is in
    # seconds. '+MaxRuntime' is mandatory on this pool: submissions without it
    # are rejected, and the '+' is part of the attribute name.
    max_runtime_s = as_int(resources.get("runtime"), DEFAULT_RUNTIME_MIN) * 60

    workdir = os.getcwd()
    # condor_submit refuses a job whose log directory does not exist.
    log_dir = os.path.join(workdir, "logs", "condor")
    os.makedirs(log_dir, exist_ok=True)
    stem = os.path.join(log_dir, f"{name}.$(CLUSTER)")

    return (
        f"universe              = vanilla\n"
        f"executable            = {jobscript}\n"
        f"initialdir            = {workdir}\n"
        f"output                = {stem}.out\n"
        f"error                 = {stem}.err\n"
        f"log                   = {stem}.log\n"
        f"getenv                = True\n"
        f"request_cpus          = {threads}\n"
        f"request_memory        = {mem_mb}\n"
        f"request_disk          = {disk_mb}\n"
        f"+MaxRuntime           = {max_runtime_s}\n"
        f'+JobBatchName         = "{name}"\n'
        f"should_transfer_files = NO\n"
        f"requirements          = {REQUIREMENTS}\n"
        f"queue\n"
    )


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: condor_submit.py <jobscript>")

    jobscript = sys.argv[1]
    properties = read_job_properties(jobscript)

    # Written alongside the jobscript in .snakemake/tmp.*/ so that a rejected
    # submission can be read back by hand.
    submit_path = jobscript + ".sub"
    with open(submit_path, "w") as submit_file:
        submit_file.write(submit_description(jobscript, properties))

    submitted = subprocess.run(
        ["condor_submit", "-terse", submit_path],
        capture_output=True,
        text=True,
    )
    if submitted.returncode != 0:
        sys.stderr.write(
            f"condor_submit failed for {job_name(properties)} "
            f"(exit {submitted.returncode}).\n"
            f"Submit description kept at: {submit_path}\n"
            f"{submitted.stdout}{submitted.stderr}"
        )
        sys.exit(1)

    # 'condor_submit -terse' prints one line per proc, e.g. '4711.0 - 4711.0'.
    # Every job here queues a single proc, so the cluster id is the whole story.
    first_line = submitted.stdout.strip().split("\n")[0]
    cluster_id = first_line.strip().split(".")[0]
    if not cluster_id.isdigit():
        sys.stderr.write(
            f"could not parse a cluster id from condor_submit output: "
            f"{submitted.stdout!r}\n"
        )
        sys.exit(1)

    print(cluster_id)


if __name__ == "__main__":
    main()
