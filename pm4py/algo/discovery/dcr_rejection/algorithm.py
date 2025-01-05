from pm4py.objects.dcr.obj import DcrGraph
from pm4py.util import exec_utils
from pm4py.algo.discovery.dcr_rejection.variants import dcr_rejection
from enum import Enum
import pandas as pd
from typing import Any, Optional, Dict

class Variants(Enum):
    DCR_REJECTION = dcr_rejection

DCR_REJECTION = Variants.DCR_REJECTION
VERSIONS = {DCR_REJECTION}


def apply(log: pd.DataFrame, variant=DCR_REJECTION, parameters: Optional[Dict[Any, Any]] = None) -> DcrGraph:
    """
    Discovers a DCR graph model from an event log, using DCR rejection mining algorithm described in [1]_.

    Parameters
    ----------
    log: pd.DataFrame
        event log as a pandas dataframe
    parameters
        Possible parameters of the algorithm, including:
        - 'activity_key' 
        - 'case_id_key' 
        - 'label_key' 
        - 'positive_label'
        - 'negative_label'
        - 'seed'
    Returns
    -------
    DcrGraph
        returns the DcrGraph mined from the log

    References
    ----------
    .. [1]
        T. Slaats, S. Debois, C. O. Back, and A. K. F. Christfort, "Foundations and practice of binary process discovery: The Rejection Miner", 
        Information Systems, vol. 121, 2024, Art. no. 102339. DOI: <https://doi.org/10.1016/j.is.2023.102339​>_.
    """

    return exec_utils.get_variant(variant).apply(log, parameters=parameters)
