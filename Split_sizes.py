"""Size of the training, validation and test samples for a given number of events.

This module is kept free of any third-party dependency, as it is imported by the Snakefile
itself (which is parsed outside of the conda environment used to run the jobs)."""

def minimum_events(percentage_train, percentage_val, max_events = 1000):
    """Smallest number of events giving a non-empty training, validation and test sample.

    The split sizes are int(percentage*n_events), so the minimum depends on the requested
    percentages: with the default 40%/40% split, 3 events are needed (2 events would give
    int(0.4*2) = 0 training and 0 validation events)."""
    for n_events in range(1, max_events + 1):
        n_train = int(percentage_train*n_events)
        n_val = int(percentage_val*n_events)
        if n_train >= 1 and n_val >= 1 and n_events - n_train - n_val >= 1:
            return n_events

    raise ValueError(
        f"No sample size up to {max_events} events can be split into a non-empty training, "
        f"validation and test sample with a training percentage of {percentage_train} and a "
        f"validation percentage of {percentage_val}. Check the percentages given."
    )
