"""Column names and topology codes shared by 'Sample_io.py' and 'ROOT_file_conv.py'.

Kept in their own module, with no dependency on either of them, so that they can each import
what they need from here without creating a circular import between the two of them."""

import json

# Name of the column holding the total weight of the events in the sample files.
WEIGHT_COLUMN = "PreWeight"
# Name of the column holding the topology code of the events. It is also the column samples are
# partitioned on when saved, so that an individual topology can be read back without scanning the
# others.
TOPOLOGY_COLUMN = "Topology"

# Mapping between the topology names (used in the config file and as the {topology} wildcard of
# the workflow) and the integer values stored in the "Topology" column.
TOPOLOGY_CODES = {
    "CC0pi": 0,
    "CC1pipm": 1,
    "CC1pi0": 2,
    "CCNpi": 3,
    "CCOther": 4,
    "NC0pi": 5,
    "NC1pipm": 6,
    "NC1pi0": 7,
    "NCNpi": 8,
    "NCOther": 9,
    "Other": 10,
}

# Total number of raw events the real downstream analysis will use for a sample, by neutrino flavour
# (the last segment of the sample path, ex: 'crpa/ND280/CH/numu' -> 'numu'), overriding
# DEFAULT_DATA_N_EVENTS. Used by 'Make_plots.py' to scale 'plot_histograms' uncertainties down to
# that many events for '1Dhist_data.pdf', instead of the much larger test-sample statistics, and by
# the metric scripts and 'Bootstrap_swd.py' to compute the data-scale metrics.
DATA_SCALE_FACTOR = 1
DEFAULT_DATA_N_EVENTS = 130400*DATA_SCALE_FACTOR
# DSF times the T2K OA2023 (runs 1-11) ND + FD event rates, the same for every detector flux and target:
#   ND: Table 4 of the OAR11B ND note (MaCh3, all weights), 130340.64 events in total. Its samples are
#       selected on reconstructed quantities, so the flavour is that of the selection: the anti-νµ
#       samples (19706.15) go to numub and every νµ sample, the RHC wrong-sign ones included
#       (110634.49), to numu. There are no νe samples.
#   FD: Tables 12-17 of the OAR11B FD note (GUNDAM post-fit, set A22), 887.26 events in total, split
#       by the flavour at interaction: numu = νµ→νµ (616.64), numub = ν̄µ→ν̄µ (129.38),
#       nue = νe→νe + νµ→νe (128.24) and nueb = ν̄e→ν̄e + ν̄µ→ν̄e (13.00).
DATA_N_EVENTS = {
    "numu": DATA_SCALE_FACTOR*(110634.49 + 616.64),
    "numub": DATA_SCALE_FACTOR*(19706.15 + 129.38),
    "nue": DATA_SCALE_FACTOR*128.24,
    "nueb": DATA_SCALE_FACTOR*13.00,
}

def sample_data_n_events(sample):
    """Return the total number of raw events the real downstream analysis will use for a sample (a
    path ending in its flavour, ex: 'crpa/ND280/CH/numu'), DEFAULT_DATA_N_EVENTS if its flavour is
    not in DATA_N_EVENTS."""
    return DATA_N_EVENTS.get(sample.rstrip("/").split("/")[-1], DEFAULT_DATA_N_EVENTS)

def data_n_events_of(total_n_events, target_topology_rates_file, topology):
    """Return the number of raw events the real downstream analysis will use for one topology: the
    total number of events of the sample, split across its topologies by the summed PreWeight of
    their target events, as written to 'target_topology_rates_file' by 'Init.py'."""
    with open(target_topology_rates_file) as f:
        topology_rates = json.load(f)
    return total_n_events * topology_rates[topology] / sum(topology_rates.values())

def topology_code(topology):
    """Return the integer code of a topology given either its name (ex: 'CC1pipm') or its code (ex: 1)."""
    if isinstance(topology, str):
        if topology in TOPOLOGY_CODES:
            return TOPOLOGY_CODES[topology]
        try:
            topology = int(topology)
        except ValueError:
            raise ValueError(f"Unknown topology '{topology}'. Please choose from {list(TOPOLOGY_CODES)}.")
    if topology not in TOPOLOGY_CODES.values():
        raise ValueError(f"Unknown topology '{topology}'. Please choose from {list(TOPOLOGY_CODES)}.")
    return int(topology)
