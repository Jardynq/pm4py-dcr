from pm4py.algo.discovery.dfg import replacement as dfg_replacement, factory as dfg_factory
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.filtering.tracelog.auto_filter import auto_filter
from pm4py.algo.filtering.tracelog.attributes import attributes_filter
from pm4py.visualization.dfg import factory as dfg_vis_factory
import os

def execute_script():
    # measure could be frequency or performance
    log_path = os.path.join("..", "tests", "input_data", "running-example.xes")
    log = xes_importer.import_log(log_path)
    filtered_log = auto_filter.apply_auto_filter(log)
    filtered_log_activities_count = attributes_filter.get_attribute_values(filtered_log, "concept:name")
    intermediate_log = attributes_filter.apply_events(log, list(filtered_log_activities_count.keys()))
    dfg_filtered_log = dfg_factory.apply(filtered_log)
    dfg_intermediate_log = dfg_factory.apply(intermediate_log)
    dfg_filtered_log = dfg_replacement.replace_values(dfg_filtered_log, dfg_intermediate_log)

    gviz = dfg_vis_factory.apply(dfg_filtered_log, log=intermediate_log, parameters={"format": "svg"})
    dfg_vis_factory.view(gviz)


if __name__ == "__main__":
    execute_script()
