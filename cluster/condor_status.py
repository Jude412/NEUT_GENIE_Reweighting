#!/usr/bin/env python3
"""Report the state of one HTCondor cluster to Snakemake.

Snakemake (7.x, legacy --cluster-status interface) invokes this as

    cluster/condor_status.py "<cluster id>"

and requires stdout to be exactly one line, one of 'running', 'success' or
'failed'. Anything else aborts the whole workflow, so every diagnostic goes to
stderr.

Why this caches
---------------
Snakemake polls on a fixed ten-second cycle and calls this command once per job
still in flight, rate-limited only by '--max-status-checks-per-second'. With a
few hundred jobs running that is hundreds of condor_q invocations a minute,
against group documentation asking for no more than one a minute. So a single
dump of the whole queue is cached on disk for CACHE_SECONDS and every job is
answered out of it. condor_history, which is much more expensive, is consulted
only for the few jobs that have just left the queue.

A note on held jobs
-------------------
A held job is reported as 'failed', which combined with '--retries' means a
transient hold costs one resubmission rather than a stalled workflow. The job
itself stays in the queue in the held state; this script deliberately does not
remove it, so that 'condor_q -hold' can still be used to find out why. Held jobs
therefore accumulate over a long run and are worth clearing by hand.
"""

# imports
import json
import os
import subprocess
import sys
import time

# How long a queue dump is reused before condor_q is called again.
CACHE_SECONDS = 30

# HTCondor JobStatus codes.
IDLE = 1
RUNNING = 2
REMOVED = 3
COMPLETED = 4
HELD = 5
TRANSFERRING_OUTPUT = 6
SUSPENDED = 7

# The states in which a job has not yet finished, one way or the other.
LIVE = frozenset({IDLE, RUNNING, TRANSFERRING_OUTPUT, SUSPENDED})


def cache_path():
    """The queue cache lives beside the Condor logs, inside the working directory."""
    return os.path.join(os.getcwd(), "logs", "condor", "queue_cache.json")


def read_cache(path):
    """The cached queue dump, or None if it is missing, stale or unreadable."""
    try:
        if time.time() - os.path.getmtime(path) >= CACHE_SECONDS:
            return None
        with open(path) as cached:
            return json.load(cached)
    except (OSError, ValueError):
        return None


def write_cache(path, states):
    """Replace the cache atomically: many copies of this script run concurrently."""
    tmp = f"{path}.{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as out:
            json.dump(states, out)
        os.replace(tmp, path)
    except OSError as error:
        # A cache we cannot write is a performance problem, not a correctness one.
        sys.stderr.write(f"could not write the queue cache: {error}\n")
        try:
            os.unlink(tmp)
        except OSError:
            pass


def queue():
    """{cluster id: [JobStatus, ExitCode]} for this user's jobs, from a cached dump."""
    path = cache_path()
    cached = read_cache(path)
    if cached is not None:
        return cached

    dump = subprocess.run(
        [
            "condor_q",
            os.environ.get("USER", ""),
            "-af",
            "ClusterId",
            "JobStatus",
            "ExitCode",
        ],
        capture_output=True,
        text=True,
    )

    states = {}
    if dump.returncode == 0:
        for line in dump.stdout.splitlines():
            fields = line.split()
            if len(fields) == 3:
                try:
                    states[fields[0]] = [int(fields[1]), fields[2]]
                except ValueError:
                    continue
    else:
        # Cache the empty result anyway, so a schedd hiccup does not turn into a
        # condor_q call from every one of the hundreds of in-flight jobs at once.
        sys.stderr.write(f"condor_q failed (exit {dump.returncode}): {dump.stderr}")

    write_cache(path, states)
    return states


def from_history(cluster_id):
    """(JobStatus, ExitCode) for a job that has left the queue, else (None, None)."""
    done = subprocess.run(
        ["condor_history", cluster_id, "-limit", "1", "-af", "JobStatus", "ExitCode"],
        capture_output=True,
        text=True,
    )
    fields = done.stdout.split()
    if len(fields) >= 2:
        try:
            return int(fields[0]), fields[1]
        except ValueError:
            pass
    return None, None


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: condor_status.py <cluster id>")
    cluster_id = sys.argv[1].strip()

    state = queue().get(cluster_id)
    if state is not None:
        status, exit_code = state
    else:
        status, exit_code = from_history(cluster_id)

    # A job can be marked COMPLETED in the queue a moment before its ExitCode is
    # populated, in which case the history is the authority.
    if status == COMPLETED and not str(exit_code).isdigit():
        status, exit_code = from_history(cluster_id)

    if status in LIVE:
        print("running")
    elif status is None:
        # Neither the queue nor the history knows this job: the history lags the
        # queue by a few seconds. Call it running and let the next poll settle it.
        print("running")
    elif status == COMPLETED and str(exit_code) == "0":
        print("success")
    else:
        # HELD, REMOVED, or completed with a non-zero exit code. For a held job,
        # 'condor_q -hold' gives the reason.
        sys.stderr.write(
            f"cluster {cluster_id}: JobStatus={status} ExitCode={exit_code}\n"
        )
        print("failed")


if __name__ == "__main__":
    main()
