from math import ceil, floor
from typing import Tuple, List, Union, Callable, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from megatron.core.pipeline_parallel.scheduler.graph import ScheduledNode, VScheduledNode, FuncType


DUALPIPE_NODETYPE = {"F", "B", "W", "Full_B", "EMPTY_BUBBLE"}


class DualPipeGraph_(object):
    """DualPipeGraph
    We brokendown DualPipe to three Pipe_Stage: Warmup, Middle, End
    Warmup contains:
        step1: no_cross_fwd
        step2: cross_fwd
        step3: warmup_1F1B1W
        step4: warmup_transitions
    Middle contains: (named as their shape in pipe)
        step1: mid_rhombic
        step2: mid_butterfly
        step3: mid_transitions
    End contains:
        step1: bwdB_step
        step2: cross_bwdB_bwdW
        step3: bwdW_step
    """

    def __init__(
        self,
        n_stage,
        n_micro,
        f_cost: int = 1,
        b_cost: int = 1,
        w_cost: int = 1,
        c_cost: int = 1,
        f_mem: int = 1,
        b_mem: int = 1,
        w_mem: int = 1,
        max_mem: int = None,
    ):
        self.n_node = 6 * n_stage * n_micro
        self.n_stage = n_stage
        self.n_micro = n_micro
        self.f_cost = f_cost
        self.b_cost = b_cost
        self.w_cost = w_cost
        self.c_cost = c_cost
        self.f_mem = f_mem
        self.b_mem = b_mem
        self.w_mem = w_mem
        self.fbw_cost = [f_cost, b_cost, w_cost]
        self.fbw_mem = [f_mem, b_mem, w_mem]
        self.max_mem = max_mem or f_mem * self.n_stage * 2

        # time unit
        self.one_time_unit = 1
        # total mbs (both up and down)
        self.total_mbs = (n_stage + 2) * 2
        # one side mbs
        self.mbs = self.total_mbs // 2

    def print_details(
        self,
        pipeline_schedule: List[List[ScheduledNode]],
        chunk_mode: bool = False,
        mbs_mode: bool = False,
        empty_bubble_str_mode: bool = False,
    ):
        assert not (
            chunk_mode and mbs_mode
        ), "Only one mode is supported at the same time, please choose from chunk_mode and mbs_mode"
        schedule_str = ""
        for stage in range(self.n_stage):
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

    def get_pipe_first_b_w(self, stage_pipe: List[ScheduledNode], chunk: int = 0):
        # get first d, last d, first u, last u B node in range[first B, first W]
        first_d, last_d, first_u, last_u = self.n_micro // 2, 0, self.n_micro // 2, 0
        stage_pipe_temp = []
        for node in stage_pipe[::-1]:
            if node.type == "Full_B":
                break
            else:
                stage_pipe_temp.append(node)
        stage_pipe = stage_pipe_temp[::-1]  # node from last fully B to ...
        # print(f"stage_pipe {[str(_.minibatch) + _.type + ('u' if _.chunk == 0 else 'd') for _ in stage_pipe]}")
        if chunk == 0:
            # get first d
            for node in stage_pipe:
                if node.type == "B" and node.chunk == 1:
                    first_d = node.minibatch
                    break

            # get first u
            for node in stage_pipe:
                if node.type == "B" and node.chunk == 0:
                    first_u = node.minibatch
                    break

            # get last_d
            for node in stage_pipe[::-1]:
                if node.type == "B" and node.chunk == 1:
                    last_d = node.minibatch
                    break

            # get last_u
            for node in stage_pipe[::-1]:
                if node.type == "B" and node.chunk == 0:
                    last_u = node.minibatch
                    break
        else:
            # get first d
            for node in stage_pipe:
                if node.type == "B" and node.chunk == 1:
                    first_d = node.minibatch
                    break

            # get first u
            for node in stage_pipe:
                if node.type == "B" and node.chunk == 0:
                    first_u = node.minibatch
                    break

            # get last_d
            for node in stage_pipe[::-1]:
                if node.type == "B" and node.chunk == 1:
                    last_d = node.minibatch
                    break

            # get last_u
            for node in stage_pipe[::-1]:
                if node.type == "B" and node.chunk == 0:
                    last_u = node.minibatch
                    break
        return first_d, last_d, first_u, last_u

    def cross_merge_nodes(
        self, node_list1: List[ScheduledNode], node_list2: List[ScheduledNode]
    ) -> List[ScheduledNode]:
        """
        corss merge node in Step: get_end_schedule-->cross_bwdB_bwdW
        example 1:
        inputs:
            node_list1:[Node 1, Node 3, Node 5]
            node_list2:[Node 2, Node 4, Node 6]
        return:
            node_list3:[Node 1, Node 2, Node 3, Node 4, Node 5, Node 6]

        example 2:
        inputs:
            node_list1:[Node 1, Node 3, Node 5]
            node_list2:[Node 2,]
        return:
            node_list3:[Node 1, Node 2, Node 3, Node 5]
        """
        merged = []
        for x, y in zip(node_list1, node_list2):
            merged.append(x)
            merged.append(y)
        merged += node_list1[len(node_list2) :]  # deal list1 rest ele
        merged += node_list2[len(node_list1) :]  # deal list2 rest ele
        return merged

    ################
    # Pipe_Stage 1
    ################
    def get_warmup_schedule(self, pipeline_schedule: List[List[ScheduledNode]]):
        ########### Pipe_Stage 1.1 ###########
        def no_cross_fwd(pipeline_schedule: List[List[ScheduledNode]]):
            # stage [0,pp/2)
            for stage in range(0, self.n_stage // 2):
                # add num i empty bubble
                start_time = 0
                for i in range(0, stage):
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="EMPTY_BUBBLE",
                            chunk=0,
                            stage=stage,
                            minibatch=0,
                            start_time=start_time,
                            completion_time=start_time + self.one_time_unit,
                        )
                    )
                    start_time += self.one_time_unit
                # add FWD node
                # Stage i in [0, pp/2)  mbs m in range [0, (pp - 1） - 2i) model chunk 0 fwd
                curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                for i in range(0, (self.n_stage - 1) - 2 * stage):
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="F",
                            chunk=0,
                            stage=stage,
                            minibatch=i,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit

            # stage [pp/2,n_stage)
            for stage in range(self.n_stage // 2, self.n_stage):
                start_time = 0
                for i in range(0, self.n_stage - stage - 1):
                    # add num i empty bubble
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="EMPTY_BUBBLE",
                            chunk=0,
                            stage=stage,
                            minibatch=0,
                            start_time=start_time,
                            completion_time=start_time + self.one_time_unit,
                        )
                    )
                    start_time += self.one_time_unit
                # add FWD node
                # Stage i in [pp/2, pp) , mbs m in range [0, 2i - (pp-1)) model chunk 1 fwd
                curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                for i in range(0, 2 * stage - (self.n_stage - 1)):
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="F",
                            chunk=1,
                            stage=stage,
                            minibatch=i,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit

        ########### Pipe_Stage 1.2 ###########
        # For each stage, add schedule Nodes col pp/2 times (range 0 to self.n_stage//2 + 1)，
        def cross_fwd(pipeline_schedule: List[List[ScheduledNode]]):
            for r in range(0, self.n_stage // 2 + 1):
                if r == 0:
                    # special case; only one col 0d 0u
                    for stage in range(r, self.n_stage // 2):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                    for stage in range(self.n_stage // 2, self.n_stage - r):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    if r % 2 != 0:
                        for stage in range(r, self.n_stage // 2):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            # Stage i in [r, pp/2) , mbs为 (pp - 1） - 2i + (r-1) model chunk 0 fwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=(self.n_stage - 1) - 2 * stage + (r - 1),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            # Stage i in [r, pp/2) , mbs r model chunk 1 fwd # 全1d
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=r,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                        for stage in range(self.n_stage // 2, self.n_stage - r):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            # Stage i in [pp/2, pp-r), mbs 为2i - (pp-1)  + (r-1) model chunk 1 fwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=2 * stage - (self.n_stage - 1) + (r - 1),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            # Stage i in [pp/2, pp-r) 压入mbs r model chunk 0 fwd # 全1u
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=r,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                    if r % 2 == 0:
                        for stage in range(r, self.n_stage // 2):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            # Stage i in [r, pp/2) , mbs为 (pp - 1） - 2i + (r-1) model chunk 0 fwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=self.n_stage - 2 * stage + (r - 2),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            # Stage i in [r, pp/2) , mbs r model chunk 1 fwd # 全1d
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=r,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                        for stage in range(self.n_stage // 2, self.n_stage - r):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            # Stage i in [pp/2, pp-r), mbs 为2i - (pp-1)  + (r-1) model chunk 1 fwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=2 * stage - (self.n_stage - 2) + (r - 2),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            # Stage i in [pp/2, pp-r) 压入mbs r model chunk 0 fwd # 全1u
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=r,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

        ########### Pipe_Stage 1.3 ###########
        def warmup_1f1b1w(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add Schedule Nodes pp/2 times from (0, self.n_stage//2 + 1)
            for r in range(0, self.n_stage // 2):
                # [0, pp/2 - r)
                for stage in range(0, self.n_stage // 2 - 1 - r):
                    ###### bwd b ######
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    # Stage i in [r, pp/2) , mbs为 (pp - 1） - 2i + (r-1) model chunk 0 fwd
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="B",
                            chunk=1,
                            stage=stage,
                            minibatch=r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                    ###### bwd w ######
                    # Stage i in [0, pp/2 - r) , mbs r model chunk 1  bwd w
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="W",
                            chunk=1,
                            stage=stage,
                            minibatch=r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                    ###### fwd ######
                    # Stage i in [0, pp/2 - r), mbs i + 1 + r model chunk 1 fwd
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="F",
                            chunk=1,
                            stage=stage,
                            minibatch=stage + 1 + r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit

                # [pp/2 + 1 + r, pp)
                for stage in range(self.n_stage // 2 + 1 + r, self.n_stage):
                    ###### bwd b ######
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    # Stage i in [0, pp/2 - r), mbs i + 1 + r model chunk 1 fwd
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="B",
                            chunk=0,
                            stage=stage,
                            minibatch=r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                    ###### bwd w ######
                    # Stage i in [pp/2 + 1 - r, pp) ,mbs r model chunk 0  bwd w
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="W",
                            chunk=0,
                            stage=stage,
                            minibatch=r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                    ###### fwd ######
                    # Stage i in [0, pp/2 - r) , mbs i + 1 + r model chunk 1 fwd
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="F",
                            chunk=0,
                            stage=stage,
                            minibatch=self.n_stage - stage + r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit

        ########### Pipe_Stage 1.4 ###########
        def warmup_transitions(pipeline_schedule: List[List[ScheduledNode]]):
            # For each stage, add pp/2 - 1 round Schedule Nodes
            for r in range(0, self.n_stage // 2):
                if r == 0:
                    # special round add 1 col fwd and 1 col fullyBwd
                    for stage in range(0, self.n_stage // 2):
                        ###### Fwd ######
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        # Stage i in [0, pp/2 ) , mbs pp - i - 1 - r model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage - stage - 1 - r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd ######
                        # Stage i in [0, pp/2) , add mbs (pp/2) - i - 1 - r model chunk 0 Full_B
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage // 2 - stage - 1 - r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2, self.n_stage):
                        ###### Fwd ######
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        # Stage i in [pp/2, pp), mbs i + r model chunk chunk 1  fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd ######
                        # Stage i in [pp/2, pp), mbs i - (pp/2) + r model chunk chunk 1 FullyB
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=stage - self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    for stage in range(0, self.n_stage // 2 - r):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        # Stage i in [0, pp/2 - r) EMPTY_BUBBLE
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="EMPTY_BUBBLE",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2 + r, self.n_stage):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        # Stage i in [pp/2 + r, pp) 压入空泡
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="EMPTY_BUBBLE",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        no_cross_fwd(pipeline_schedule)
        cross_fwd(pipeline_schedule)
        warmup_1f1b1w(pipeline_schedule)
        warmup_transitions(pipeline_schedule)

    ################
    # Pipe_Stage 2
    ################
    def get_middle_schedule(self, pipeline_schedule: List[List[ScheduledNode]]):
        ########### Pipe_Stage 2.1 ###########
        def mid_rhombic(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add (pp/2) + 1 round（total 9 round）Schedule Nodes
            for r in range(0, self.n_stage // 2 + 1):
                if r == 0:
                    for stage in range(0, self.n_stage // 2):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd 1 ######
                        # Stage i in [0, pp/2 ), mbs pp/2 model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd 1 ######
                        # Stage i in [0, pp/2) , mbs r model chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd 2 ######
                        # Stage i in [0, pp/2 ) , mbs pp - i -r model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage - stage - r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully Bwd 2 ######
                        # Stage i in [0, pp/2) , mbs (pp/2) - i - r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 - stage - r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2, self.n_stage):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd 1 ######
                        # Stage i in [pp/2, pp), mbs pp/2 + r model chunk chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd 1 ######
                        # Stage i in [0, pp/2) , mbs r model chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd 2 ######
                        # Stage i in [pp/2, pp) , mbs i + 1 + r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=stage + 1 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd 2 ######
                        # Stage i in [pp/2, pp) , mbs i - ((pp/2) -1) + r model chunk chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=stage - (self.n_stage // 2 - 1) + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    for stage in range(r - 1, self.n_stage // 2):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd 1 ######
                        # Stage i in [r - 1, pp/2), mbs (pp/2) + r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully Bwd 1 ######
                        # Stage i in [r - 1, pp/2), mbs r model chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd 2 ######
                        # Stage i in [r - 1, pp/2) , mbs pp + r - i model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage + r - stage,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully Bwd 2 ######
                        # Stage i in [r - 1, pp/2), mbs (pp/2) - i + r model chunk 1 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 - stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2, self.n_stage - r + 1):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd 1 ######
                        # Stage i in [pp/2, pp - r + 1) , mbs (pp/2) + r model chunk chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully Bwd 1 ######
                        # Stage i in [pp/2, pp - r + 1), mbs r model chunk chunk 1 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd 2 ######
                        # Stage i in [pp/2, pp - r + 1), mbs i + 1 + r model chunk chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=stage + 1 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd 2 ######
                        # Stage i in [pp/2, pp - r + 1), mbs i - ((pp/2) -1) + r model chunk chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=stage - (self.n_stage // 2 - 1) + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        ########### Pipe_Stage 2.2 ###########
        def mid_butterfly(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add pp/2 round（total 8 round）Schedule Nodes
            for r in range(0, self.n_stage // 2 + 1):
                if r == 0:
                    for stage in range(0, self.n_stage // 2 - r):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd ######
                        # Stage i in [0, pp/2 - r ), mbs bs//2-pp//2+i+r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_micro // 2 - self.n_stage // 2 + stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2 + r, self.n_stage):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd ######
                        # Stage i in [pp/2 + r, pp), mbs (bs//2 + pp//2 - 1)- i + r model chunk chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_micro // 2 + self.n_stage // 2 - 1 - stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    for stage in range(0, self.n_stage // 2 - r):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fully bwd 1 ######
                        # Stage i in [0, pp/2 - r ), mbs i + r + 1 model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=stage + r + 1,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Empty Bubble ######
                        # Stage i in [0, pp/2 - r) , Empty Bubble
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="EMPTY_BUBBLE",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully bwd 2 ######
                        # Stage i in [0, pp/2 - r), mbs bs//2-pp//2 - 1 + r model chunk 1 Fully bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_micro // 2 - self.n_stage // 2 - 1 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd ######
                        # Stage i in [0, pp/2 - r), mbs bs//2 - pp//2 + i + r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_micro // 2 - self.n_stage // 2 + stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                    for stage in range(self.n_stage // 2 + r, self.n_stage):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fully bwd 1 ######
                        # Stage i in [pp/2 + r, pp), mbs bs - i + r -2 model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage - stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Empty Bubble ######
                        # Stage i in [pp/2 + r, pp), Empty Bubble
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="EMPTY_BUBBLE",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully bwd 2 ######
                        # Stage i in [pp/2 + r, pp), mbs bs//2-pp//2 - 1 + r model chunk 0 Fully bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_micro // 2 - self.n_stage // 2 - 1 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd ######
                        # Stage i in [pp/2 + r, pp), mbs (bs/2+pp/2 - 1) - i + r model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_micro // 2 + self.n_stage // 2 - 1 - stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        ########### Pipe_Stage 2.3 ###########
        def mid_transitions(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add pp/2 + 1 round（total 9 round）Schedule Nodes
            for r in range(0, self.n_stage // 2 + 1):
                if r == 0:
                    for stage in range(r, self.n_stage // 2):
                        ###### Fully B ######
                        # Stage i in [r, pp/2), mbs pp/2 + r + 1  model chunk 0 fully bwd
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r + 1,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                    for stage in range(self.n_stage // 2, self.n_stage):
                        ###### Fully B ######
                        # Stage i in [pp/2, pp), mbs pp/2 + r + 1 model chunk 1 fully bwd
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r + 1,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    if r % 2 != 0:  # odd round: 1, 3, 5, 7
                        for stage in range(r - 1, self.n_stage // 2):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            ###### Empty Bubble ######
                            # Stage i in [r - 1, pp/2), Empty bubble
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="EMPTY_BUBBLE",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=0,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            ###### Fully bwd ######
                            # Stage i in [r - 1, pp/2), mbs pp + r - i model chunk 1 fully bwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="Full_B",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=self.n_stage + ceil(r / 2) - stage,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                        for stage in range(self.n_stage // 2, self.n_stage - r + 1):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            ###### Empty Bubble ######
                            # Stage i in [pp/2, pp - r + 1), Empty bubble
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="EMPTY_BUBBLE",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=0,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            ###### Fully bwd ######
                            # Stage i in [pp/2, pp - r + 1)  压入mbs i + 1 + r model chunk chunk 0  fully bwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="Full_B",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=stage + 1 + ceil(r / 2),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                    else:  # even round: 2, 4, 6, 8
                        for stage in range(r - 1, self.n_stage // 2):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            ###### Empty Bubble ######
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="EMPTY_BUBBLE",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=0,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            ###### Fully bwd ######
                            # Stage i in [r - 1, pp/2), mbs pp//2 + r model chunk 0 fully bwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="Full_B",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=self.n_stage // 2 + floor(r / 2) + 1,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                        for stage in range(self.n_stage // 2, self.n_stage - r + 1):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            ###### Empty Bubble ######
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="EMPTY_BUBBLE",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=0,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            ###### Fully bwd ######
                            # Stage i in [pp/2, pp - r + 1), mbs  pp//2 + r model chunk chunk 1 fully bwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="Full_B",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=self.n_stage // 2 + floor(r / 2) + 1,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

        mid_rhombic(pipeline_schedule)
        mid_butterfly(pipeline_schedule)
        mid_transitions(pipeline_schedule)

    ################
    # Pipe_Stage 3
    ################
    def get_end_schedule(self, pipeline_schedule: List[List[ScheduledNode]]):
        ########### Pipe_Stage 3.1 ###########
        def bwdB_step(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, pp/2 round（total 8 round）Schedule Nodes，
            for r in range(0, self.n_stage // 2):
                if r % 2 == 0:
                    # Stage i in [pp/2 - r - 1, pp/2)
                    for stage in range(self.n_stage // 2 - r - 1, self.n_stage // 2):
                        # Stage i in [pp/2 - r - 1, pp/2), mbs (pp/2 - 1）*2 + r/2 model chunk 1  bwd B
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="B",
                                chunk=1,
                                stage=stage,
                                minibatch=(self.n_stage // 2 - 1) * 2 + r // 2,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    # Stage i in [pp/2, pp/2 + r + 1)
                    for stage in range(self.n_stage // 2, self.n_stage // 2 + r + 1):
                        # Stage i in [pp/2, pp/2 + r+ 1), mbs (pp/2 - 1）*2 + r/2 model chunk 0 bwd B
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="B",
                                chunk=0,
                                stage=stage,
                                minibatch=(self.n_stage // 2 - 1) * 2 + r // 2,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    # Stage i in [pp/2 - r - 1, pp/2)
                    for stage in range(self.n_stage // 2 - r - 1, self.n_stage // 2):
                        # Stage i in [pp/2 - r - 1, pp/2), mbs pp-1-(pp/2 - i)+ floor(r/2)  model chunk 0 bwd B  # [6:13,7:14]
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="B",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage - 1 - (self.n_stage // 2 - stage) + floor(r / 2),
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    # Stage i in [pp/2, pp/2 + r+ 1)
                    for stage in range(self.n_stage // 2, self.n_stage // 2 + r + 1):
                        # Stage i in [pp/2, pp/2 + r+ 1), mbs pp-1-(i - pp/2 + 1) + floor(r/2) model chunk 1 bwd B # [8:14,9:13]
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage - 1 - (stage - self.n_stage // 2 + 1) + floor(r / 2),
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        ########### Pipe_Stage 3.2 ###########
        def cross_bwdB_bwdW(pipeline_schedule: List[List[ScheduledNode]]):
            for stage in range(0, self.n_stage // 2 - 1):
                first_d, last_d, first_u, last_u = self.get_pipe_first_b_w(pipeline_schedule[stage], chunk=0)
                # print(f"stage {stage} Up first_d {first_d}, last_d {last_d}, first_u {first_u}, last_u {last_u} ")
                u_queue_w, u_queue_b, d_queue_w = [], [], []
                ### 1.Get W nodes, then merge up/down W nodes ###
                # get up W nodes: [first_u: mbs//2]
                for _ in range(first_u, self.n_micro // 2):
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    u_queue_w.append(
                        ScheduledNode(
                            type="W",
                            chunk=0,
                            stage=stage,
                            minibatch=_,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                # get down W nodes: [first_d: mbs//2] Bwd W to W Queue
                for _ in range(first_d, self.n_micro // 2):
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    d_queue_w.append(
                        ScheduledNode(
                            type="W",
                            chunk=1,
                            stage=stage,
                            minibatch=_,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                ### 2.Get B nodes, then cross with W ###
                for _ in range(last_u, self.n_micro // 2):
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    u_queue_b.append(
                        ScheduledNode(
                            type="B",
                            chunk=0,
                            stage=stage,
                            minibatch=_ + 1,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                # if stage % 2 == 0: u_queue_w first, then d_queue_w
                if stage % 2 == 0:
                    w_nodes = self.cross_merge_nodes(u_queue_w, d_queue_w)
                    wb_nodes = self.cross_merge_nodes(w_nodes, u_queue_b)
                    # clean w nodes, let it stop at mbs // 2 - 1, chunk 0, type 'B'
                    cut_idx = len(wb_nodes)
                    for _ in range(len(wb_nodes)):
                        if (
                            wb_nodes[_].minibatch == (self.n_micro // 2 - 1)
                            and wb_nodes[_].type == "B"
                            and wb_nodes[_].chunk == 0
                        ):
                            cut_idx = _
                            break
                    wb_nodes = wb_nodes[: cut_idx + 1]
                    # append nodes to stage pipe
                    pipeline_schedule[stage].extend(wb_nodes)
                    # print(f"stage {stage} cut_idx {cut_idx} wb_nodes {[str(_.minibatch) + _.type + ('u' if _.chunk == 0 else 'd') for _ in wb_nodes]}")
                # else: d_queue_w first, then u_queue_w
                else:
                    w_nodes = self.cross_merge_nodes(d_queue_w, u_queue_w)
                    wb_nodes = self.cross_merge_nodes(w_nodes, u_queue_b)
                    # clean w nodes, let it stop at mbs // 2 - 1, chunk 0, type 'B'
                    cut_idx = len(wb_nodes)
                    for _ in range(len(wb_nodes)):
                        if (
                            wb_nodes[_].minibatch == (self.n_micro // 2 - 1)
                            and wb_nodes[_].type == "B"
                            and wb_nodes[_].chunk == 0
                        ):
                            cut_idx = _
                            break
                    wb_nodes = wb_nodes[: cut_idx + 1]
                    # append nodes to stage pipe
                    pipeline_schedule[stage].extend(wb_nodes)
                    # print(f"stage {stage} cut_idx {cut_idx} wb_nodes {[str(_.minibatch) + _.type + ('u' if _.chunk == 0 else 'd') for _ in wb_nodes]}")

            for stage in range(self.n_stage // 2 + 1, self.n_stage):
                first_d, last_d, first_u, last_u = self.get_pipe_first_b_w(pipeline_schedule[stage], chunk=1)
                # print(f"stage {stage} Down first_d {first_d}, last_d {last_d}, first_u {first_u}, last_u {last_u} ")
                d_queue_w, d_queue_b, u_queue_w = [], [], []
                ### 1.Get W nodes, then merge down/up W nodes ###
                # get down W nodes: [first_d: mbs // 2] chunk 1
                for _ in range(first_d, self.n_micro // 2):
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    d_queue_w.append(
                        ScheduledNode(
                            type="W",
                            chunk=1,
                            stage=stage,
                            minibatch=_,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                # print(f"stage {stage} d_queue_w {[_.minibatch for _ in d_queue_w]}")
                # get up W nodes: [first_u: mbs//2] chunk 0
                for _ in range(first_u, self.n_micro // 2):
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    u_queue_w.append(
                        ScheduledNode(
                            type="W",
                            chunk=0,
                            stage=stage,
                            minibatch=_,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                # print(f"stage {stage} u_queue_w {[_.minibatch for _ in u_queue_w]}")
                ### 2.Get B nodes, then cross with W ###
                for _ in range(last_d, self.n_micro // 2):
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    d_queue_b.append(
                        ScheduledNode(
                            type="B",
                            chunk=1,
                            stage=stage,
                            minibatch=_ + 1,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                # print(f"stage {stage} d_queue_b {[_.minibatch for _ in d_queue_b]}")
                # print(
                #     f"stage {stage} d_queue_w {[_.minibatch for _ in d_queue_w]} d_queue_b {[_.minibatch for _ in d_queue_b]} u_queue_w {[_.minibatch for _ in u_queue_w]}"
                # )
                if stage % 2 == 0:
                    w_nodes = self.cross_merge_nodes(u_queue_w, d_queue_w)
                    # print(f"stage {stage} w_nodes {[_.minibatch for _ in w_nodes]}")
                    wb_nodes = self.cross_merge_nodes(w_nodes, d_queue_b)
                    # clean w nodes, let it stop at mbs // 2 - 1, chunk 1, type 'B'
                    cut_idx = len(wb_nodes)
                    for _ in range(len(wb_nodes)):
                        if (
                            wb_nodes[_].minibatch == (self.n_micro // 2 - 1)
                            and wb_nodes[_].type == "B"
                            and wb_nodes[_].chunk == 1
                        ):
                            cut_idx = _
                            break
                    wb_nodes = wb_nodes[: cut_idx + 1]
                    # append nodes to stage pipe
                    pipeline_schedule[stage].extend(wb_nodes)
                    # print(f"stage {stage} cut_idx {cut_idx} wb_nodes {[str(_.minibatch) + _.type + ('u' if _.chunk == 0 else 'd') for _ in wb_nodes]}")
                else:
                    w_nodes = self.cross_merge_nodes(d_queue_w, u_queue_w)
                    # print(f"stage {stage} w_nodes {[_.minibatch for _ in w_nodes]}")
                    wb_nodes = self.cross_merge_nodes(w_nodes, d_queue_b)
                    # clean w nodes, let it stop at mbs // 2 - 1, chunk 1, type 'B'
                    cut_idx = len(wb_nodes)
                    for _ in range(len(wb_nodes)):
                        if (
                            wb_nodes[_].minibatch == (self.n_micro // 2 - 1)
                            and wb_nodes[_].type == "B"
                            and wb_nodes[_].chunk == 1
                        ):
                            cut_idx = _
                            break
                    wb_nodes = wb_nodes[: cut_idx + 1]
                    # append nodes to stage pipe
                    pipeline_schedule[stage].extend(wb_nodes)
                    # print(f"stage {stage} cut_idx {cut_idx} wb_nodes {[str(_.minibatch) + _.type + ('u' if _.chunk == 0 else 'd') for _ in wb_nodes]}")
                # # else: d_queue_w first, then u_queue_w
                # else:
                #     w_nodes = self.cross_merge_nodes(u_queue_w, d_queue_w)
                #     wb_nodes = self.cross_merge_nodes(w_nodes, d_queue_b)
                #     # clean w nodes, let it stop at mbs // 2 - 1, chunk 0, type 'B'
                #     cut_idx = len(wb_nodes)
                #     for _ in range(len(wb_nodes)):
                #         if (
                #             wb_nodes[_].minibatch == (self.n_micro // 2 - 1)
                #             and wb_nodes[_].type == "B"
                #             and wb_nodes[_].chunk == 1
                #         ):
                #             cut_idx = _
                #             break
                #     wb_nodes = wb_nodes[: cut_idx + 1]
                #     print(f"stage {stage} cut_idx {cut_idx} wb_nodes {[str(_.minibatch) + _.type + ('u' if _.chunk == 0 else 'd') for _ in wb_nodes]}")

        ########### Pipe_Stage 3.3 ###########
        def bwdW_step(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add pp/2 round（total 8 round）Schedule Nodes
            for r in range(0, self.n_stage // 2):
                for stage in range(0, self.n_stage // 2):
                    # Red up # [0, 7]
                    if stage in range(self.n_stage // 2 - r - 1, self.n_stage // 2 - 1):
                        # Stage i in [pp/2 - r, pp/2 - 1), mbs(pp/2 - 1）*2 + 向下取整(r/2) + 1 model chunk Chunk_num   bwd W  # None
                        # mbs_num = (pp/2 - 1）*2 + 向下取整(r/2)  if r < pp //2 // 2 else  (pp/2 - 1）*2 + (r -  pp //2 // 2)
                        # chunk_num   = 1 if r < pp //2 else 0
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        chunk_num = 1 if r < self.n_stage // 2 // 2 else 0
                        mbs_num = (
                            (self.n_stage // 2 - 1) * 2 + r
                            if r < self.n_stage // 2 // 2
                            else (self.n_stage // 2 - 1) * 2 + (r - self.n_stage // 2 // 2)
                        )
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="W",
                                chunk=chunk_num,
                                stage=stage,
                                minibatch=mbs_num,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    # Blue up [7, 8]
                    if stage in range(self.n_stage // 2 - 1, self.n_stage // 2):
                        # Stage i in [pp/2 - 1, pp/2), mbs (pp/2 - 1）*2 + floor(r/2) model chunk 1  bwd W
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="W",
                                chunk=1 if r % 2 == 0 else 0,
                                stage=stage,
                                minibatch=(self.n_stage // 2 - 1) * 2 + floor(r / 2),
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                for stage in range(self.n_stage // 2, self.n_stage):
                    # Blue down [8, 9]
                    if stage in range(self.n_stage // 2, self.n_stage // 2 + 1):
                        # Stage i in [pp/2, pp/2 + 1), mbs (pp/2 - 1）*2 + 向下取整(r/2) - 1 model chunk chunk 0  bwd W
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="W",
                                chunk=0 if r % 2 == 0 else 1,
                                stage=stage,
                                minibatch=(self.n_stage // 2 - 1) * 2 + floor(r / 2),
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                    # Red Down # [9, 16]
                    if stage in range(self.n_stage // 2 + 1, self.n_stage // 2 + r + 1):
                        # Stage i in [pp/2 + 1, pp/2 + r+ 1), mbs(pp/2 - 1）*2 + floor(r/2)  model chunk  0  bwd W
                        # mbs_num = (pp/2 - 1）*2 + 向下取整(r/2)  if r < pp //2 // 2 else  (pp/2 - 1）*2 + (r -  pp //2 // 2)
                        # chunk_num   = 1 if r < pp //2 else 0
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        chunk_num = 0 if r < self.n_stage // 2 // 2 else 1
                        mbs_num = (
                            (self.n_stage // 2 - 1) * 2 + r
                            if r < self.n_stage // 2 // 2
                            else (self.n_stage // 2 - 1) * 2 + (r - self.n_stage // 2 // 2)
                        )
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="W",
                                chunk=chunk_num,
                                stage=stage,
                                minibatch=mbs_num,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        bwdB_step(pipeline_schedule)
        cross_bwdB_bwdW(pipeline_schedule)
        bwdW_step(pipeline_schedule)

    def get_dualpipe_schedule(
        self,
    ):
        pipeline_schedule = [[] for _ in range(self.n_stage)]
        self.get_warmup_schedule(pipeline_schedule)
        self.get_middle_schedule(pipeline_schedule)
        self.get_end_schedule(pipeline_schedule)
        return pipeline_schedule
    

# Reference from https://github.com/deepseek-ai/DualPipe/blob/main/dualpipe/dualpipe.py
class DualPipeGraph(object):
    def __init__(
        self,
        pp_size: int = 1, 
        num_microbatch: int = 1, 
    ) -> None:
        self.pp_size = pp_size
        self.num_microbatch = num_microbatch
        self.dualpipe_schedules: List[List[VScheduledNode]] = [[] for _ in range(pp_size)]
        
        # rank_mapping: Map rank in process_group to actual pp rank.
        # rank_inverse_mapping: Map actual pp rank to rank in process_group.
        # if rank_mapping is None:
        self.rank_mapping = list(range(self.pp_size))
        self.rank_inverse_mapping = [None] * (self.pp_size + 1)
        for i in range(self.pp_size):
            self.rank_inverse_mapping[self.rank_mapping[i]] = i
        
        self.current_f_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_b_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_w_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_send_f_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_send_b_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_recv_f_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_recv_b_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
    
    def get_first_rank(self,):
        return self.rank_inverse_mapping[0]
    
    def get_last_rank(self,):
        return self.rank_inverse_mapping[self.pp_size - 1]
    
    def get_prev_rank(self, curr_rank):
        return self.rank_inverse_mapping[curr_rank - 1]
    
    def get_next_rank(self, curr_rank):
        return self.rank_inverse_mapping[curr_rank + 1]
    
    def is_first_rank(self, curr_rank):
        return curr_rank == 0
    
    def is_last_rank(self, curr_rank):
        return curr_rank == self.pp_size - 1
    
    def is_in_second_half(self, curr_rank):
        return curr_rank >= self.pp_size // 2
    
    def is_middle_rank(self, curr_rank):
        return (curr_rank == self.pp_size // 2 - 1) or (curr_rank == self.pp_size // 2)

    
    def forward_compute_schedule(self, phase: int, rank:int, microbatch_id: int = None) -> None:
        phase ^= self.is_in_second_half(curr_rank=rank)
        chunk_id = self.current_f_chunk_id[rank][phase]

        is_last_stage = (self.is_first_rank(curr_rank=rank) and phase == 1) or (self.is_last_rank(curr_rank=rank) and phase == 0)
        self.dualpipe_schedules[rank].append(
            VScheduledNode(
                type=FuncType.F,
                chunk=phase,
                stage=rank,
                minibatch=chunk_id,
                start_time=0,
                completion_time=0, 
            )
        )
        self.current_f_chunk_id[rank][phase] += 1

    def backward_compute_schedule(self, phase: int, rank:int, microbatch_id: int = None, enable_zb: bool = False) -> None:
        phase ^= self.is_in_second_half(curr_rank=rank)
        chunk_id = self.current_b_chunk_id[rank][phase]
        self.dualpipe_schedules[rank].append(
            VScheduledNode(
                type=FuncType.B if enable_zb else FuncType.BW,
                chunk=phase,
                stage=rank,
                minibatch=chunk_id,
                start_time=0,
                completion_time=0, 
            )
        )
        self.current_w_chunk_id[rank][phase] += 1
        
        self.current_b_chunk_id[rank][phase] += 1

        is_last_stage = (self.is_first_rank(curr_rank=rank) and phase == 1) or (self.is_last_rank(curr_rank=rank) and phase == 0)


    def forward_backward_compute_schedule(self, phase0: int, phase1: int, rank:int, fwd_microbatch_id: int, bwd_microbatch_id: int,) -> None:
        self.forward_compute_schedule(phase=phase0, rank=rank, microbatch_id=self.current_f_chunk_id[rank][phase0])
        self.backward_compute_schedule(phase=phase1, rank=rank, microbatch_id=self.current_b_chunk_id[rank][phase1])

    def forward_schedule(self, phase: int, rank: int, microbatch_id: int = None, recv: bool = True, send: bool = True) -> None:
        if recv:
            self.recv_forward_schedule(phase=phase, rank=rank, microbatch_id=microbatch_id)
        # self._commit_and_wait_comm()

        self.forward_compute_schedule(phase=phase, rank=rank, microbatch_id=microbatch_id)

        if send:
            self.send_forward_schedule(phase=phase, rank=rank, microbatch_id=microbatch_id)

    def backward_schedule(self, phase: int, rank:int, microbatch_id: int = None, enable_zb: bool = False, recv: bool = True, send: bool = True) -> None:
        if recv:
            self.recv_backward_schedule(phase=phase, rank=rank, microbatch_id=microbatch_id)

        self.backward_compute_schedule(phase=phase, rank=rank, microbatch_id=microbatch_id, enable_zb=enable_zb)

        if send:
            self.send_backward_schedule(phase=phase, rank=rank, microbatch_id=microbatch_id)

    def forward_backward_schedule(self, phase0: int, phase1: int, rank:int, fwd_microbatch_id: int = None, bwd_microbatch_id: int = None, recv0: bool = True) -> None:
        if recv0:
            self.recv_forward_schedule(phase0, rank, fwd_microbatch_id,)
        self.recv_backward_schedule(phase1, rank, bwd_microbatch_id,)

        self.forward_backward_compute_schedule(phase0, phase1, rank, fwd_microbatch_id, bwd_microbatch_id)

        self.send_forward_schedule(phase0, rank, fwd_microbatch_id,)
        self.send_backward_schedule(phase1, rank, bwd_microbatch_id,)

    def weight_compute_schedule(self, phase: int, rank:int, microbatch_id: int=None,) -> None:
        phase ^= self.is_in_second_half(curr_rank=rank)
        chunk_id = microbatch_id if microbatch_id else self.current_w_chunk_id[rank][phase]
        
        self.dualpipe_schedules[rank].append(
            VScheduledNode(
                type=FuncType.W,
                chunk=phase,
                stage=rank,
                minibatch=chunk_id,
                start_time=0,
                completion_time=0, 
            )
        )
        self.current_w_chunk_id[rank][phase] += 1

    def recv_forward_schedule(self, phase: int, rank: int, microbatch_id: int) -> None:
        phase ^= self.is_in_second_half(curr_rank=rank)
        is_first_stage = (self.is_first_rank(curr_rank=rank) and phase == 0) or (self.is_last_rank(curr_rank=rank) and phase == 1)
        if is_first_stage:
            return
        self.dualpipe_schedules[rank].append(
            VScheduledNode(
                type=FuncType.RECV_FORWARD,
                chunk=phase,
                stage=rank,
                minibatch=self.current_recv_f_chunk_id[rank][phase],
                start_time=0,
                completion_time=0, 
            )
        )

        self.current_recv_f_chunk_id[rank][phase] += 1


    def send_forward_schedule(self, phase: int, rank:int, microbatch_id: int) -> None:
        phase ^= self.is_in_second_half(curr_rank=rank)
        is_last_stage = (self.is_first_rank(curr_rank=rank) and phase == 1) or (self.is_last_rank(curr_rank=rank) and phase == 0)
        if is_last_stage:
            return
        chunk_id = self.current_send_f_chunk_id[rank][phase]
        self.dualpipe_schedules[rank].append(
            VScheduledNode(
                type=FuncType.SEND_FORWARD,
                chunk=phase,
                stage=rank,
                minibatch=chunk_id,
                start_time=0,
                completion_time=0, 
            )
        )

        self.current_send_f_chunk_id[rank][phase] += 1

    def recv_backward_schedule(self, phase: int, rank:int, microbatch_id: int) -> None:
        phase ^= self.is_in_second_half(curr_rank=rank)
        is_last_stage = (self.is_first_rank(curr_rank=rank) and phase == 1) or (self.is_last_rank(curr_rank=rank) and phase == 0)
        if is_last_stage:
            return
        self.dualpipe_schedules[rank].append(
            VScheduledNode(
                type=FuncType.RECV_BACKWARD,
                chunk=phase,
                stage=rank,
                minibatch=self.current_recv_b_chunk_id[rank][phase],
                start_time=0,
                completion_time=0, 
            )
        )
        self.current_recv_b_chunk_id[rank][phase] += 1


    def send_backward_schedule(self, phase: int, rank:int, microbatch_id: int) -> None:
        phase ^= self.is_in_second_half(curr_rank=rank)
        is_first_stage = (self.is_first_rank(curr_rank=rank) and phase == 0) or (self.is_last_rank(curr_rank=rank) and phase == 1)
        if is_first_stage:
            return

        chunk_id = self.current_send_b_chunk_id[rank][phase]
        
        self.dualpipe_schedules[rank].append(
            VScheduledNode(
                type=FuncType.SEND_BACKWARD,
                chunk=phase,
                stage=rank,
                minibatch=chunk_id,
                start_time=0,
                completion_time=0, 
            )
        )
        self.current_send_b_chunk_id[rank][phase] += 1


    def get_dualpipe_schedule(
        self, 
        additional_warmup: int = 0,
    ) -> List[List[ScheduledNode]]:
        """
        """
        additional_warmup = additional_warmup
        for rank in range(self.pp_size):
            num_ranks = self.pp_size
            num_chunks = self.num_microbatch
            assert num_ranks % 2 == 0
            assert num_chunks > 0 and num_chunks % 2 == 0 and num_chunks >= num_ranks * 2, f"{num_chunks=}, {num_ranks=}"
            num_half_ranks = num_ranks // 2
            half_rank = min(rank, num_ranks - 1 - rank)
            half_num_chunks = num_chunks // 2

            # For the first half of the ranks: phase 0 means forward direction, phase 1 means reverse direction.
            # For the second half of the ranks: phase 0 means reverse direction, phase 1 means forward direction.

            # Step 1: nF0
            # no addition warmup: step_1 = (num_half_ranks - half_rank - 1) * 
            step_1 = (num_half_ranks - half_rank - 1) * 2
            for i in range(step_1):
                self.forward_schedule(phase=0,rank=rank, microbatch_id=None)

            # Step 2: nF0F1
            ############## we add one more warmup step for both phase 0 and phase 1 ##############
            step_2 = half_rank + 1 + additional_warmup 
            self.recv_forward_schedule(phase=0, rank=rank, microbatch_id=None)
            for i in range(step_2):
                self.forward_schedule(phase=0, rank=rank, microbatch_id=None, recv=False, send=self.is_middle_rank(curr_rank=rank))
                self.recv_forward_schedule(phase=0, rank=rank, microbatch_id=None,)
                self.forward_schedule(phase=1, rank=rank, microbatch_id=None, send=(not self.is_middle_rank(curr_rank=rank)) or (i < step_2 - 1))
                if not self.is_middle_rank(curr_rank=rank):
                    self.send_forward_schedule(phase=0, rank=rank, microbatch_id=None)

            # Step 3: nB1W1F1 (Use zero bubble)
            step_3 = num_half_ranks - half_rank - 1
            for i in range(step_3):
                self.backward_schedule(phase=1, rank=rank, microbatch_id=None, enable_zb=True)
                self.recv_forward_schedule(phase=1, rank=rank, microbatch_id=None,)
                self.weight_compute_schedule(phase=0, rank=rank, microbatch_id=None,)
                self.forward_schedule(phase=1, rank=rank, microbatch_id=None,recv=False)

            # # Step 4 (Main step): nF0B1F1B0
            step_4 = half_num_chunks - num_ranks + half_rank + 1
            ############## we reduce one warmup step for F0 and F1 ##############
            for i in range(step_4):
                # F0B1
                if i == 0:
                    if self.is_middle_rank(curr_rank=rank):
                        # NOTE: We don't overlap these two chunks to further reduce bubble size.
                        self.forward_schedule(phase=0, rank=rank, microbatch_id=None, recv=False, send=False)
                        self.send_forward_schedule(phase=1, rank=rank, microbatch_id=None,)
                        self.backward_schedule(phase=1, rank=rank, microbatch_id=None, send=False)
                        self.send_forward_schedule(phase=0, rank=rank, microbatch_id=None,)
                        self.send_backward_schedule(phase=1, rank=rank, microbatch_id=None,)
                    else:
                        self.forward_backward_schedule(phase0=0, phase1=1, rank=rank, recv0=False)
                else:
                    if i < step_4 - additional_warmup:
                        # Do F0B1
                        self.forward_backward_schedule(phase0=0, phase1=1, rank=rank, )
                    else:
                        # Do B1 only
                        self.backward_schedule(phase=1, rank=rank, microbatch_id=None)
                    
                    # self.forward_backward_schedule(phase0=0, phase1=1, rank=rank, )
                    

                if i < step_4 - additional_warmup:
                    # Do F1B0
                    self.forward_backward_schedule(phase0=1, phase1=0, rank=rank,)
                else:
                    # Do B0 only
                    self.backward_schedule(phase=0, rank=rank, microbatch_id=None)
                
                # self.forward_backward_schedule(phase0=1, phase1=0, rank=rank, )

            # # Step 5: nB1F1B0
            step_5 = num_half_ranks - half_rank - 1
            ############## we reduce one warmup step for F1 ##############
            for i in range(step_5):
                self.backward_schedule(phase=1, rank=rank)
                self.forward_backward_schedule(phase0=1, phase1=0, rank=rank)

            # Step 6: nB1B0 (The second half of the chunks use zero bubble)
            step_6 = half_rank + 1
            enable_zb = False
            for i in range(step_6):
                if i == step_6 // 2 and half_rank % 2 == 1:
                    enable_zb = True
                self.backward_schedule(phase=1, rank=rank, enable_zb=enable_zb)
                if i == step_6 // 2 and half_rank % 2 == 0:
                    enable_zb = True
                self.backward_schedule(phase=0, rank=rank, enable_zb=enable_zb)

            # Step 7: nWB0 (Use zero bubble)
            step_7 = num_half_ranks - half_rank - 1
            
            phase0_last_BW_microbatch_id = 0
            phase1_last_BW_microbatch_id = 0
            for node in self.dualpipe_schedules[rank]:
                if node.type == FuncType.BW:
                    if node.chunk == 0:
                        phase0_last_BW_microbatch_id = node.minibatch
                    elif node.chunk == 1:
                        phase1_last_BW_microbatch_id = node.minibatch
            phase0_rest_W = [i for i in range(phase0_last_BW_microbatch_id + 1, (self.num_microbatch) // 2)]
            phase1_rest_W = [i for i in range(phase1_last_BW_microbatch_id + 1, (self.num_microbatch) // 2)]
            
            # # print(f"rank {rank} phase0_rest_W {phase0_rest_W} phase1_rest_W {phase1_rest_W} self.current_w_chunk_id {self.current_w_chunk_id[rank]}")
            
            for i in range(step_7):
                phase = 0
                if phase1_rest_W:
                    phase = 1
                    microbatch_id = phase1_rest_W.pop(0)
                else:
                    if phase0_rest_W:
                        phase = 0
                        microbatch_id = phase0_rest_W.pop(0)
                self.weight_compute_schedule(phase=phase, rank=rank, microbatch_id=microbatch_id)
                self.backward_schedule(phase=0, rank=rank, microbatch_id=microbatch_id, enable_zb=True)

            # Step 8: nW
            step_8 = half_rank + 1
            for i in range(step_8):
                phase = 0
                if phase1_rest_W:
                    # phase = 1
                    microbatch_id = phase1_rest_W.pop(0)
                else:
                    if phase0_rest_W:
                        # phase = 0
                        microbatch_id = phase0_rest_W.pop(0)
                self.weight_compute_schedule(phase=0, rank=rank, microbatch_id=microbatch_id)
            # print(f"rank {rank} phase0_rest_W {phase0_rest_W} phase1_rest_W {phase1_rest_W} self.current_w_chunk_id {self.current_w_chunk_id[rank]}")

    def print_pipeline_details(
        self,
        pipeline_schedule: List[List[ScheduledNode]],
        chunk_mode: bool = False,  # print model chunk
        mbs_mode: bool = False, # print mbs id
        show_comm: bool = False, # print communication ops
    ):
        assert not (
            chunk_mode and mbs_mode
        ), f"Only one mode is supported at the same time, please choose from chunk_mode {chunk_mode} and mbs_mode {mbs_mode}"
        schedule_str = ""
        
        node_type_list = [FuncType.F, FuncType.B, FuncType.W, FuncType.BW]
        if show_comm:
            node_type_list += [FuncType.SEND_FORWARD, FuncType.RECV_FORWARD, FuncType.SEND_BACKWARD, FuncType.RECV_BACKWARD]
        for stage in range(len(pipeline_schedule)):
            stage_nodes = []
            for node in pipeline_schedule[stage]:
                if node.type in node_type_list:
                    if chunk_mode:
                        stage_nodes.append(str(node.type) + str(node.chunk))
                    elif mbs_mode:
                        stage_nodes.append(str(node.type) + str(node.minibatch))
                    else:
                        stage_nodes.append(str(node.type))
            stage_str = "".join([_ for _ in stage_nodes])
            schedule_str += "\n" + stage_str
        print(schedule_str)

