"""Column names and topology codes shared by 'Sample_io.py' and 'ROOT_file_conv.py'.

Kept in their own module, with no dependency on either of them, so that they can each import
what they need from here without creating a circular import between the two of them."""

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
