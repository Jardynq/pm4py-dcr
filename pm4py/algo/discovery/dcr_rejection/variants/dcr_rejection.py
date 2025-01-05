from copy import deepcopy
import random
from enum import Enum
import pandas as pd

from pm4py.stats import get_event_attribute_values
from pm4py.objects.dcr.obj import dcr_template
from pm4py.objects.dcr.obj import DcrGraph


class MinimizeObjective(Enum):
    Naive = 1,
    Greedy = 2,


def apply(log, parameters = None) -> DcrGraph:
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
    disc = Rejection()
    return disc.mine(log, parameters = parameters)


class Rejection:
    def __init__(self):
        self.graph = deepcopy(dcr_template)
        self.oracle = deepcopy(dcr_template)
        self.oracle_iter = None
        self.events = None,
        self.positive = None,
        self.negative = None,

    def mine(self, log: pd.DataFrame, parameters=None) -> DcrGraph:
        """
        Method used for calling the underlying mining algorithm used for discovery of DCR Graphs

        Parameters
        ----------
        log
            an event log as EventLog or pandas.DataFrame

        parameters
                activity_key: optional parameter, used to identify the activities in the log
                case_id_key: optional parameter, used to identify the cases executed in the log

        Returns
        -------
        Tuple[DcrGraph,Dict[str, Any]]
            returns a tuple containing:
            - The DCR Graph
            - The log abstraction used for mining
        """
        self.init(log, parameters)
        self.minimize(MinimizeObjective.Naive)
        self.minimize(MinimizeObjective.Greedy)
        return DcrGraph(self.graph)

    def init(self, log, parameters):
        activity_key = parameters['activity_key'] 
        case_id_key = parameters['case_id_key'] 
        label_key = parameters['label_key']
        positive_label = parameters['positive_label']
        negative_label = parameters['negative_label']
        seed = parameters['seed']

        self.events = set(get_event_attribute_values(log, activity_key))

        # Extract positive and negative traces from dataframe
        # Positive and negative are dicts of the form {trace id: [event 0, ... event n]} 
        positive = log[log[label_key] == positive_label]
        self.positive = positive.groupby(case_id_key)[activity_key].apply(list).to_dict()
        negative = log[log[label_key] == negative_label]
        self.negative = negative.groupby(case_id_key)[activity_key].apply(list).to_dict()

        # Make flower model
        self.graph['marking']['included'] = deepcopy(self.events)
        self.graph['events'] = deepcopy(self.events)
        self.graph['labels'] = self.graph['events']
        self.graph['labelMapping'] = {event: event for event in self.graph['events']}

        # Create oracle
        self.oracle['marking']['included'] = deepcopy(self.events)

        # Techincally oracle relations response and exclude should be inverted,
        # but since oracle covers all relations, they are identical to their inverse.
        relations = {event: self.events - set(event) for event in self.events}
        relations_self = {event: self.events for event in self.events}
        self.oracle['conditionsFor'] = relations
        self.oracle['responseTo'] = relations
        self.oracle['excludesTo'] = relations_self

        # Oracle iter is a list of all individual relation tuples
        # Each tuple is of the form (relation_type, (event_from, event_to))
        relations_iter = [(a, b) for a in self.events for b in self.events if a != b]
        relations_self_iter = [(a, b) for a in self.events for b in self.events]
        self.oracle_iter = (
            [('conditionsFor', x) for x in relations_iter] + 
            [('responseTo', x) for x in relations_iter] + 
            [('excludesTo', x) for x in relations_self_iter]
        )
        random.Random(seed).shuffle(self.oracle_iter)

    def execute(self, graph, marking, event):
        # Current event was just executed
        marking['executed'].add(event)
        marking['pending'].discard(event)

        # Update markings of all events that were affected by current event
        marking['pending'] |= graph['responseTo'].get(event, set())
        marking['included'] -= graph['excludesTo'].get(event, set())
        marking['included'] |= graph['includesTo'].get(event, set())

    def update_mappings(self, event, exe_since_inc, exe_since_exe):
        # Clear all exSiIn events that were just included by current event
        for other in self.graph['includesTo'].get(event, set()):
            exe_since_inc[other] = set()

        # Add current event to all previous executed exSiEx events 
        # Add current event to all previous included exSiIn events 
        for other in self.events:
            exe_since_exe.setdefault(other, set()).add(event)
            exe_since_inc.setdefault(other, set()).add(event)

        # Clear current exSiEx event, since it was just executed 
        exe_since_exe[event] = set()

    def trace_rejection(self, traces):
        # responseTo and excludesTo are inverted.
        # the naming is just kept consistent for simpler code
        rejections = {
            'conditionsFor': {},
            'responseTo': {},
            'excludesTo': {},
        }

        for id, trace in traces.items():
            gm = deepcopy(self.graph['marking'])
            om = deepcopy(self.oracle['marking'])

            exe_since_inc = {}
            exe_since_exe = {}
            for event in trace:
                self.execute(self.graph, gm, event)
                self.execute(self.oracle, om, event)

                relation = self.oracle['conditionsFor'][event] & (gm['included'] - gm['executed'])
                for other in relation:
                    rejections['conditionsFor'].setdefault((event, other), set()).add(id)

                if event not in om['included']:
                    relation = self.oracle['excludesTo'][event] & exe_since_inc.get(event, set())
                    for other in relation:
                        rejections['excludesTo'].setdefault((other, event), set()).add(id)
                
                self.update_mappings(event, exe_since_inc, exe_since_exe)
            
            for event in gm['included'] & om['pending']:
                relation = self.oracle['responseTo'][event] & exe_since_exe.get(event, set())
                for other in relation:
                    rejections['responseTo'].setdefault((other, event), set()).add(id)

        return rejections

    def minimize(self, objective):
        while self.negative:
            pos_rejects = self.trace_rejection(self.positive)
            get_pos = lambda x: pos_rejects[x[0]].get(x[1], set())
            len_pos = lambda x: len(get_pos(x))

            neg_rejects = self.trace_rejection(self.negative)
            get_neg = lambda x: neg_rejects[x[0]].get(x[1], set())
            len_neg = lambda x: len(get_neg(x))

            if objective == MinimizeObjective.Naive:
                # Only consider relations that reject no positive traces
                # Choose the single relation that rejects the most negative traces
                best = max(
                    filter(lambda x: not get_pos(x), self.oracle_iter), 
                    key=len_neg,
                    default=None
                )
                if best is None: break
                if not get_neg(best): break
            elif objective == MinimizeObjective.Greedy:
                # Choose the single relation that rejects relatively 
                # the most negative traces and the fewest positive traces
                best = max(
                    self.oracle_iter, 
                    key=lambda x: len_neg(x) - len_pos(x),
                    default=None
                )
                if best is None: break
                if len_pos(best) >= len_neg(best): break
            else:
                raise ValueError("Unknown MinimizeObjective variant")
                
            # A relation worked
            type, (event, other) = best
            self.graph[type].setdefault(deepcopy(event), set()).add(deepcopy(other))
            keep_pos = set(self.positive.keys()) - get_pos(best)
            keep_neg = set(self.negative.keys()) - get_neg(best)
            self.positive = {id: self.positive[id] for id in keep_pos}
            self.negative = {id: self.negative[id] for id in keep_neg}
