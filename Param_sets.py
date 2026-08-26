"""Named sets of analysis parameters the samples and the metrics are built for.

The workflow computes its samples and its metrics for several sets of parameters (the '3D' and
'8D' sets of the previous analyses, the 'all' set holding every analysis parameter, and the set
of parameters the models are trained on). They are listed in the 'parameters' section of the
config file rather than hard-coded, so that adding or removing a set is a config file change.

A set is given on the command line of the scripts as '--param_set NAME param1 param2 ...', and
can be given as many times as there are sets. The helpers below turn those arguments into
dictionaries, and give the columns of a set in a sample holding every analysis parameter."""

# imports
import argparse

# Name of the set holding every analysis parameter. It is always built, as the metrics of the
# other sets are computed from the samples holding every parameter.
ALL_PARAMS_SET = "all"

def add_param_set_argument(argparser, help_suffix=""):
    """Add the repeatable '--param_set' argument to a parser."""
    argparser.add_argument("--param_set", action="append", nargs="+", required=False, default=[],
                           metavar=("NAME", "PARAM"),
                           help="Name of a set of analysis parameters followed by the parameters it holds "
                                f"(ex: '--param_set 3D Enu_true PLep CosLep'). Can be given once per set.{help_suffix}")

def param_sets_from_args(param_set_args):
    """Return the {set name: list of parameters} dictionary of the '--param_set' arguments."""
    param_sets = {}
    for param_set in param_set_args or []:
        name, params = param_set[0], param_set[1:]
        if not params:
            raise argparse.ArgumentTypeError(f"No parameter given for the parameter set '{name}'.")
        if name in param_sets:
            raise argparse.ArgumentTypeError(f"The parameter set '{name}' is given more than once.")
        param_sets[name] = list(params)
    return param_sets

def named_paths_from_args(path_args):
    """Return the {set name: path} dictionary of a repeatable 'NAME path' argument."""
    named_paths = {}
    for name, path in path_args or []:
        if name in named_paths:
            raise argparse.ArgumentTypeError(f"A path is given more than once for the parameter set '{name}'.")
        named_paths[name] = path
    return named_paths

def indices_of(params, analysis_params):
    """Return the columns holding the given parameters in a sample holding every analysis parameter."""
    missing = [param for param in params if param not in analysis_params]
    if missing:
        raise ValueError(f"The parameters {missing} are not held by the analysis parameters {list(analysis_params)}. "
                         "Every parameter of a set of the config file must also be listed in its "
                         "'parameters'/'all' section.")
    return [list(analysis_params).index(param) for param in params]
