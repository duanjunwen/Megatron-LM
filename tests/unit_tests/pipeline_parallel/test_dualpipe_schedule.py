from typing import List
# from megatron.core.pipeline_parallel.scheduler.dualpipe_schedule import DUALPIPE_NODETYPE, DualPipeGraph
from megatron.core.pipeline_parallel.scheduler.dualpipe_schedule import DUALPIPE_NODETYPE, DualPipeGraph
from megatron.core.pipeline_parallel.scheduler.v_schedule import DualVPipelineGraph, PipelineGraph, ScheduledNode
from utils import parameterize


def print_pipeline_details(
    pipeline_schedule: List[List[ScheduledNode]],
    chunk_mode: bool = False,  # print model chunk
    mbs_mode: bool = False, # print mbs id
    empty_bubble_str_mode: bool = False, # print Empty bubble
    show_comm: bool = False, # print communication ops
):
    assert not (
        chunk_mode and mbs_mode
    ), "Only one mode is supported at the same time, please choose from chunk_mode and mbs_mode"
    schedule_str = ""
    for stage in range(len(pipeline_schedule)):
        stage_nodes = []
        for node in pipeline_schedule[stage]:
            if node.type in DUALPIPE_NODETYPE:
                if node.type == "EMPTY_BUBBLE":
                    if empty_bubble_str_mode:
                        stage_nodes.append("E")
                    else:
                        stage_nodes.append(" ")
                else:
                    if chunk_mode:
                        stage_nodes.append(node.type + str(node.chunk))
                    elif mbs_mode:
                        stage_nodes.append(node.type + str(node.minibatch))
                    else:
                        stage_nodes.append(node.type)
        stage_str = "".join([_ for _ in stage_nodes])
        schedule_str += "\n" + stage_str
    print(schedule_str)


@parameterize(
    "test_config",
    [
        {
            "n_stage": 8,
            "additional_warmup":1,
        },
    ],
)
def test_dualpipe_schedule(test_config):
    dualpipe = DualPipeGraph(
        pp_size=test_config["n_stage"], 
        num_microbatch=20,
    )
    dualpipe.get_dualpipe_schedule(
        additional_warmup=test_config['additional_warmup']
    )
    print(f"\nDualPipe V schedule\n")
    dualpipe.print_pipeline_details(
        pipeline_schedule=dualpipe.dualpipe_schedules,
        show_comm=False,
        mbs_mode=True,
        # chunk_mode=True,
    )
    
@parameterize(
    "test_config",
    [
        {
            "n_stage": 4,
        },
    ],
)
def test_zerobubbleV_schedule(test_config):
    mem_f = 34 * 4096 + 5 * 24 * 4096
    mem_w = - 32 * 4096
    mem_b = - mem_w - mem_f
    # zbv
    zbv_schedule = PipelineGraph(
        n_stage=test_config["n_stage"],
        n_micro=8,
        f_cost=1,
        b_cost=1,
        w_cost=1,
        c_cost=1,
        f_mem=mem_f * 1.5,
        b_mem=mem_b * 1.5,
        w_mem=mem_w * 1.5,
    ).get_v_schedule()
    print(f"\nZeroBubble V schedule\n")
    print_pipeline_details(
        zbv_schedule,
        mbs_mode=True,
        empty_bubble_str_mode=True
    )


@parameterize(
    "test_config",
    [
        {
            "n_stage": 4,
            "additional_warmup":1,
        },
    ],
)
def test_dualpipeV_schedule(test_config):
    # dual V
    dualV_graph = DualVPipelineGraph(
        pp_size=test_config["n_stage"], 
        num_microbatch=10, 
    )
    dualV_graph.get_dual_v_schedule(
        additional_warmup=test_config['additional_warmup']
    )
    
    print(f"\nDualPipe V schedule\n")
    dualV_graph.print_pipeline_details(
        pipeline_schedule=dualV_graph.dualV_schedules,
        show_comm=False,
        mbs_mode=True,
        # chunk_mode=True,
    )
    

if __name__ == "__main__":
    test_dualpipe_schedule()
    # test_zerobubbleV_schedule()
    test_dualpipeV_schedule()