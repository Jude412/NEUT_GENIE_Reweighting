#!/bin/bash
# properties = {properties}
#
# Custom Snakemake jobscript for HTCondor.
#
# Snakemake formats this file with str.format, so EVERY literal brace must be
# doubled. Only '{properties}' above and '{exec_job}' at the bottom are
# placeholders; a single brace anywhere else raises a WorkflowError complaining
# that a value was not found in the custom jobscript.
#
# The command in {exec_job} already begins with the absolute path of the venv
# interpreter (Snakemake writes sys.executable into it), so no environment needs
# activating here.

set -o pipefail

# Temporary files go to the node's local scratch disk - at least 20 GB, and much
# faster than CephFS for the many small files parquet writing produces.
# Snakemake defers evaluating its 'tmpdir' resource until the job actually runs,
# so exporting TMPDIR here is enough for it to be picked up.
export TMPDIR="${{_CONDOR_SCRATCH_DIR:-/tmp}}"

# HTCondor sets OMP_NUM_THREADS to the number of requested CPUs, but leaves the
# other threading libraries alone. Left to themselves each starts one thread per
# physical core, which is how a job ends up using far more CPU than it requested
# and gets held for it.
export OPENBLAS_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"

# One line of provenance per job, into the job's .err file.
echo "[jobscript] host=$(hostname) cpus=${{OMP_NUM_THREADS:-?}} tmpdir=${{TMPDIR}}" >&2

{exec_job}
