def get_outgoing_edges(dfg):
    """
    Gets outgoing edges of the provided DFG graph
    """
    outgoing = {}
    for el in dfg:
        if type(el[0]) is str:
            if not el[0] in outgoing:
                outgoing[el[0]] = {}
            outgoing[el[0]][el[1]] = dfg[el]
        else:
            if not el[0][0] in outgoing:
                outgoing[el[0][0]] = {}
            outgoing[el[0][0]][el[0][1]] = el[1]
    return outgoing


def get_ingoing_edges(dfg):
    """
    Get ingoing edges of the provided DFG graph
    """
    ingoing = {}
    for el in dfg:
        if type(el[0]) is str:
            if not el[1] in ingoing:
                ingoing[el[1]] = {}
            ingoing[el[1]][el[0]] = dfg[el]
        else:
            if not el[0][1] in ingoing:
                ingoing[el[0][1]] = {}
            ingoing[el[0][1]][el[0][0]] = el[1]
    return ingoing


def infer_start_activities(dfg):
    """
    Infer start activities from a Directly-Follows Graph

    Parameters
    ----------
    dfg
        Directly-Follows Graph

    Returns
    ----------
    start_activities
        Start activities in the log
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)

    start_activities = []

    for act in outgoing:
        if not act in ingoing:
            start_activities.append(act)

    return start_activities


def infer_end_activities(dfg):
    """
    Infer end activities from a Directly-Follows Graph

    Parameters
    ----------
    dfg
        Directly-Follows Graph

    Returns
    ----------
    end_activities
        End activities in the log
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)

    end_activities = []

    for act in ingoing:
        if not act in outgoing:
            end_activities.append(act)

    return end_activities


def get_activities_from_dfg(dfg):
    """
    Get the list of attributes directly from DFG graph

    Parameters
    -----------
    dfg
        Directly-Follows graph

    Returns
    -----------
    list_activities
        List of activities that are present in the DFG graph
    """
    set_activities = set()
    for el in dfg:
        if type(el[0]) is str:
            set_activities.add(el[0])
            set_activities.add(el[1])
        else:
            set_activities.add(el[0][0])
            set_activities.add(el[0][1])
    list_activities = sorted(list(set_activities))

    return list_activities


def get_max_activity_count(dfg, act):
    """
    Get maximum count of an ingoing/outgoing edge related to an activity

    Parameters
    ------------
    dfg
        Directly-Follows graph
    act
        Activity

    Returns
    ------------
    max_value
        Maximum count of ingoing/outgoing edges to attributes
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)
    max_value = -1
    if act in ingoing:
        for act2 in ingoing[act]:
            if ingoing[act][act2] > max_value:
                max_value = ingoing[act][act2]
    if act in outgoing:
        for act2 in outgoing[act]:
            if outgoing[act][act2] > max_value:
                max_value = outgoing[act][act2]
    return max_value


def sum_ingoutg_val_activ(dictio, activity):
    """
    Gets the sum of ingoing/outgoing values of an activity

    Parameters
    -----------
    dictio
        Dictionary
    activity
        Current examined activity

    Returns
    -----------
    summ
    """
    summ = 0
    for act2 in dictio[activity]:
        summ += dictio[activity][act2]
    return summ


def max_occ_all_activ(dfg):
    """
    Get maximum ingoing/outgoing sum of values related to attributes in DFG graph
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)
    max_value = -1

    for act in ingoing:
        summ = sum_ingoutg_val_activ(ingoing, act)
        if summ > max_value:
            max_value = summ

    for act in outgoing:
        summ = sum_ingoutg_val_activ(outgoing, act)
        if summ > max_value:
            max_value = summ

    return max_value


def max_occ_among_specif_activ(dfg, activities):
    """
    Get maximum ingoing/outgoing sum of values related to attributes in DFG graph
    (here attributes to consider are specified)
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)
    max_value = -1

    for act in activities:
        if act in ingoing:
            summ = sum_ingoutg_val_activ(ingoing, act)
            if summ > max_value:
                max_value = summ
        if act in outgoing:
            summ = sum_ingoutg_val_activ(outgoing, act)
            if summ > max_value:
                max_value = summ

    return max_value


def sum_start_activities_count(dfg):
    """
    Gets the sum of start attributes count inside a DFG

    Parameters
    -------------
    dfg
        Directly-Follows graph

    Returns
    -------------
        Sum of start attributes count
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)

    sum_values = 0

    for act in outgoing:
        if not act in ingoing:
            for act2 in outgoing[act]:
                sum_values += outgoing[act][act2]

    return sum_values


def sum_end_activities_count(dfg):
    """
    Gets the sum of end attributes count inside a DFG

    Parameters
    -------------
    dfg
        Directly-Follows graph

    Returns
    -------------
        Sum of start attributes count
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)

    sum_values = 0

    for act in ingoing:
        if not act in outgoing:
            for act2 in ingoing[act]:
                sum_values += ingoing[act][act2]

    return sum_values


def sum_activities_count(dfg, activities):
    """
    Gets the sum of specified attributes count inside a DFG

    Parameters
    -------------
    dfg
        Directly-Follows graph
    activities
        Activities to sum

    Returns
    -------------
        Sum of start attributes count
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)

    sum_values = 0

    for act in activities:
        if act in outgoing:
            for act2 in outgoing[act]:
                sum_values += outgoing[act][act2]
        if act in ingoing:
            for act2 in ingoing[act]:
                sum_values += ingoing[act][act2]
        if act in ingoing and act in outgoing:
            sum_values = int(sum_values / 2)

    return sum_values


def filter_dfg_on_act(dfg, listact):
    """
    Filter a DFG graph on a list of attributes
    (to produce a projected DFG graph)

    Parameters
    -----------
    dfg
        Current DFG graph
    listact
        List of attributes to filter on
    """
    new_dfg = []
    for el in dfg:
        if el[0][0] in listact and el[0][1] in listact:
            new_dfg.append(el)
    return new_dfg


def negate(dfg):
    """
    Negate relationship in the DFG graph

    Parameters
    ----------
    dfg
        Directly-Follows graph

    Returns
    ----------
    negated_dfg
        Negated Directly-Follows graph (for parallel cut detection)
    """
    negated_dfg = []

    outgoing = get_outgoing_edges(dfg)

    for el in dfg:
        if not (el[0][1] in outgoing and el[0][0] in outgoing[el[0][1]]):
            negated_dfg.append(el)

    return negated_dfg

def get_activities_direction(dfg, activities):
    """
    Calculate activities direction (in a similar way to Heuristics Miner)

    Parameters
    -----------
    dfg
        Directly-follows graph
    activities
        (if provided) activities of the subtree

    Returns
    -----------
    direction
        Dictionary that contains for each direction a number that goes from -1 (all ingoing edges)
        to 1 (all outgoing edges)
    """

    if activities is None:
        activities = get_activities_from_dfg(dfg)

    ingoing_list = get_ingoing_edges(dfg)
    outgoing_list = get_outgoing_edges(dfg)

    direction = {}
    for act in activities:
        outgoing = 0
        ingoing = 0
        if act in outgoing_list:
            outgoing = sum(list(outgoing_list[act].values()))
        if act in ingoing_list:
            ingoing = sum(list(ingoing_list[act].values()))
        dependency = (outgoing - ingoing) / (ingoing + outgoing + 1)
        direction[act] = dependency
    return direction

def get_activities_dirlist(activities_direction):
    """
    Form an ordered list out of a dictionary that contains for each activity
    the direction (going from -1 if all ingoing edges, to 1 if all outgoing edges)

    Parameters
    -----------
    activities_direction
        Dictionary that contains for each direction a number that goes from -1 (all ingoing edges)
        to 1 (all outgoing edges)

    Returns
    ----------
    dirlist
        Sorted list of couples of activity plus the direction
    """
    dirlist = []
    for act in activities_direction:
        dirlist.append([act, activities_direction[act]])
    dirlist = sorted(dirlist, key=lambda x: (x[1], x[0]), reverse=True)
    return dirlist


def get_activities_self_loop(dfg):
    """
    Get attributes that are in self-loop in this subtree

    Parameters
    ----------
    dfg
        Directly-follows graph

    Returns
    ----------
    self_loop_act
        Activities of the graph that are in subloop
    """
    self_loop_act = []

    outgoing = get_outgoing_edges(dfg)

    for act in outgoing:
        if act in list(outgoing[act].keys()):
            self_loop_act.append(act)
    return self_loop_act
