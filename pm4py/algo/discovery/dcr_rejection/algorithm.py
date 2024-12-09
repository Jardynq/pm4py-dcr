from pm4py.objects.log.obj import EventLog
from pm4py.util import exec_utils
from pm4py.algo.discovery.dcr_rejection.variants import dcr_rejection
from enum import Enum, auto
import pandas as pd
from typing import Union, Any, Optional, Dict, Tuple, Set

class Variants(Enum):
    DCR_REJECTION = dcr_rejection

DCR_REJECTION = Variants.DCR_REJECTION
VERSIONS = {DCR_REJECTION}


def apply(log: Union[EventLog, pd.DataFrame], variant=DCR_REJECTION, findAdditionalConditions: bool = True,
          post_process: Optional[Set[str]] = None, parameters: Optional[Dict[Any, Any]] = None) -> Tuple[Any, dict]:
    """
    discover a DCR graph from a provided event log, implemented the DisCoveR algorithm presented in [1]_.
    Allows for mining for additional attribute currently implemented mining of organisational attributes.

    Parameters
    ---------------
    log: EventLog | pd.DataFrame
        event log used for discovery
    variant
        Variant of the algorithm to use:
        - DCR_BASIC
    findAdditionalConditions:
        Parameter determining if the miner should include an extra step of mining for extra conditions
        - [True, False]

    post_process
        kind of post process mining to handle further patterns
        - DCR_ROLES

    parameters
        variant specific parameters
        findAdditionalConditions: [True or False]

    Returns
    ---------------
    DcrGraph | DistributedDcrGraph | HierarchicalDcrGraph | TimeDcrGraph:
        DCR graph (as an object) containing eventId, set of activities, mapping of event to activities,
            condition relations, response relation, include relations and exclude relations.
        possible to return variant of different dcr graph depending on which variant, basic, distributed, etc.

    References
    ----------
    .. [1]
        C. O. Back et al. "DisCoveR: accurate and efficient discovery of declarative process models",
        International Journal on Software Tools for Technology Transfer, 2022, 24:563–587. 'DOI' <https://doi.org/10.1007/s10009-021-00616-0>_.
    """

    input_log = log  # deepcopy(log)
    graph, la = exec_utils.get_variant(variant).apply(input_log, findAdditionalConditions=findAdditionalConditions, parameters=parameters)

    if post_process is None:
        post_process = set()

    return graph, la
