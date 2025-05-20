# Refer from Zero Bubble Pipeline Parallelism.
# Github: https://github.com/sail-sg/zero-bubble-pipeline-parallelism
# Paper: https://arxiv.org/abs/2401.10241
# The following applies to all files unless otherwise noted:
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections import deque
import copy
from dataclasses import dataclass
from typing import List
from megatron.core.pipeline_parallel.scheduler.graph import ScheduledNode, VScheduledNode, FuncType


class PipelineGraph(object):
    """PipelineGraph"""

    def __init__(
        self,
        n_stage,
        n_micro,
        f_cost,
        b_cost,
        w_cost,
        c_cost,
        f_mem,
        b_mem,
        w_mem,
        max_mem=None,
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

    def get_id(self, cat, chunk, stage, micro):
        return (
            cat * 2 * self.n_stage * self.n_micro + chunk * self.n_stage * self.n_micro + stage * self.n_micro + micro
        )

    def try_v_schedule(self, fill_f=True, fill_b=True, approved_bubble=None):
        count = []
        for i in range(self.n_stage):
            count.append([0] * 6)

        end_time = [-1] * self.n_node
        cur_time = [0] * self.n_stage
        mem = [0] * self.n_stage
        stage_bubble = [0] * self.n_stage
        pending_w = [deque() for _ in range(self.n_stage)]
        schedule = [[] for _ in range(self.n_stage)]
        stage_str = ["    " * i for i in range(self.n_stage)]

        if approved_bubble is None:
            approved_bubble = [-1] * self.n_stage
        max_approved_bubble = max(approved_bubble)

        def get_max_stage_bubble(stage=-1):
            max_stage_bubble = 0
            for bb in stage_bubble:
                max_stage_bubble = max(max_stage_bubble, bb)
            if stage >= 0:
                max_stage_bubble = max(max_stage_bubble, max_approved_bubble - approved_bubble[stage])
            return max_stage_bubble

        def put_w(stage):
            assert len(pending_w[stage]) > 0
            _, chunk_, _ = pending_w[stage].popleft()
            put(2, chunk_, stage)

        def put(cat, chunk, stage, assert_cnt=True):
            _tmp = _no_bubble = cur_time[stage] + self.fbw_cost[cat]
            _cnt = count[stage][cat * 2 + chunk]
            # assert _cnt < self.n_micro
            if _cnt >= self.n_micro:
                if not assert_cnt:
                    stage_str[stage] += "    "
                    cur_time[stage] = _tmp  # TODO
                    return
                assert False
            assert mem[stage] + self.fbw_mem[cat] <= self.max_mem
            stage_str[stage] += "FfBbWw"[cat * 2 + chunk] + str(_cnt + 1) + " " * (3 - len(str(_cnt + 1)))
            if cat > 0 or chunk > 0:
                last_id = cat * 2 + chunk - 1
                if cat < 2:
                    assert end_time[self.get_id(last_id // 2, last_id % 2, stage, _cnt)] >= 0
                else:
                    assert end_time[self.get_id(1, chunk, stage, _cnt)] >= 0
            if chunk == 1 and cat < 2:
                if stage < self.n_stage - 1:
                    _fa_id = self.get_id(cat, chunk, stage + 1, _cnt)
                    assert end_time[_fa_id] >= 0
                    _tmp = max(_tmp, end_time[_fa_id] + self.c_cost + self.fbw_cost[cat])
            if chunk == 0 and cat < 2:
                if stage > 0:
                    _fa_id = self.get_id(cat, chunk, stage - 1, _cnt)
                    assert end_time[_fa_id] >= 0, f"{cat}, {chunk}, {stage}, {_cnt}"
                    _tmp = max(_tmp, end_time[_fa_id] + self.c_cost + self.fbw_cost[cat])
            _id = self.get_id(cat, chunk, stage, _cnt)
            if count[stage][0] > 0:
                stage_bubble[stage] += _tmp - _no_bubble
            end_time[_id] = _tmp
            cur_time[stage] = _tmp
            mem[stage] += self.fbw_mem[cat]
            # noinspection PyTypeChecker
            schedule[stage].append((cat, chunk, _cnt))
            if cat == 1:
                pending_w[stage].append((2, chunk, _cnt))
            count[stage][cat * 2 + chunk] += 1

        for i in range(self.n_stage):
            put(0, 0, i)
        for i in range(self.n_stage - 1, -1, -1):
            if i == self.n_stage - 1:
                put(0, 1, i)
                continue
            tmp = end_time[self.get_id(0, 1, i + 1, 0)] + self.c_cost
            while (
                mem[i] + self.fbw_mem[0] * (2 + i * 2) <= self.max_mem
                and cur_time[i] + self.fbw_cost[0] <= tmp
                and count[i][0] < self.n_micro
            ):
                for j in range(i + 1):
                    put(0, 0, j)
            put(0, 1, i)
        iter_chunk_ = 0
        end_tmp = 0
        for i in range(self.n_stage):
            if i == 0:
                end_tmp = cur_time[0] + self.fbw_cost[1]
                continue
            tmp = end_tmp + self.c_cost
            while (
                count[i][0] + count[i][1] < count[i - 1][0] + count[i - 1][1]
                or count[i][1] <= count[i - 1][1] < self.n_micro
            ):
                for j in range(self.n_stage - 1, i - 1, -1):
                    if count[j][iter_chunk_] < self.n_micro:
                        put(0, iter_chunk_, j)
                iter_chunk_ = 1 - iter_chunk_

        for _ in range(2 * self.n_micro):
            # check mem before putting b
            for i in range(self.n_stage):
                while mem[i] + self.fbw_mem[1] > self.max_mem:
                    assert len(pending_w[i]) > 0
                    put_w(i)
            b0_ranks, b1_ranks = [], []
            for i in range(self.n_stage):
                if count[i][3] >= count[i][2]:
                    b0_ranks.append(i)
                elif i == self.n_stage - 1:
                    b1_ranks.append(i)
                else:
                    fa_id = self.get_id(1, 1, i + 1, count[i][3])
                    if end_time[fa_id] >= 0 or count[i][2] >= self.n_micro:
                        b1_ranks.append(i)
                    else:
                        b0_ranks.append(i)
            b_ranks = []
            # put b1
            for i in reversed(b1_ranks):
                b_ranks.append((i, 1))
            # put b0
            for i in b0_ranks:
                b_ranks.append((i, 0))
            for i, _chunk_ in b_ranks:
                fa_id = -1
                if _chunk_ == 1 and i < self.n_stage - 1:
                    fa_id = self.get_id(1, 1, i + 1, count[i][3])
                if _chunk_ == 0 and i > 0:
                    fa_id = self.get_id(1, 0, i - 1, count[i][2])
                while (
                    len(pending_w[i]) > 0
                    and fa_id >= 0
                    and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]
                ):
                    # fill the bubble
                    put_w(i)
                if (
                    len(pending_w[i]) > 0
                    and end_time[fa_id] + self.c_cost - cur_time[i] > get_max_stage_bubble(i) - stage_bubble[i]
                ):
                    if _chunk_ == 1:
                        put_w(i)
                    elif fill_b:
                        put_w(i)
                put(1, _chunk_, i)

            # put f
            for i in range(self.n_stage):
                if count[i][1] >= self.n_micro:
                    continue
                put_item = None
                if count[i][1] >= count[i][0]:
                    put_item = 0
                elif i == self.n_stage - 1:
                    put_item = 1
                else:
                    if end_time[self.get_id(0, 1, i + 1, count[i][1])] >= 0:
                        put_item = 1
                    elif count[i][0] < self.n_micro:
                        if i == 0:
                            put_item = 0
                        elif end_time[self.get_id(0, 0, i - 1, count[i][0])] >= 0:
                            put_item = 0
                if put_item is None:
                    continue
                # check mem before putting f
                while mem[i] + self.fbw_mem[0] > self.max_mem:
                    assert len(pending_w[i]) > 0
                    put_w(i)
                fa_id = -1
                if put_item == 0 and i > 0:
                    fa_id = self.get_id(0, 0, i - 1, count[i][0])
                if put_item == 1 and i < self.n_stage - 1:
                    fa_id = self.get_id(0, 1, i + 1, count[i][1])
                while (
                    len(pending_w[i]) > 0
                    and fa_id >= 0
                    and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]
                ):
                    # fill the bubble
                    put_w(i)
                if (
                    len(pending_w[i]) > 0
                    and end_time[fa_id] + self.c_cost - cur_time[i] > get_max_stage_bubble(i) - stage_bubble[i]
                ):
                    if fill_f:
                        put_w(i)
                put(0, put_item, i)

        for i in range(self.n_stage):
            while len(pending_w[i]) > 0:
                put_w(i)

        max_bubble = get_max_stage_bubble()
        expected_time = sum(self.fbw_cost) * self.n_micro * 2
        max_bubble / expected_time
        if max_approved_bubble < 0 or max_bubble < max_approved_bubble:
            _schedule, _end_time, _max_bubble = self.try_v_schedule(
                fill_f=fill_f,
                fill_b=fill_b,
                approved_bubble=stage_bubble,
            )
            if _max_bubble < max_bubble:
                return _schedule, _end_time, _max_bubble
        return schedule, end_time, max_bubble

    def print_details(self, end_time, print_scaling=1):
        for stage in range(self.n_stage):
            stage_str = ["."] * int(max(end_time) / print_scaling)
            for _cat in range(3):
                for _chunk in range(2):
                    for _micro in range(self.n_micro):
                        _id = self.get_id(_cat, _chunk, stage, _micro)
                        if end_time[_id] < 0:
                            continue
                        end = int(end_time[_id] / print_scaling)
                        start = int((end_time[_id] - self.fbw_cost[_cat]) / print_scaling)
                        for j in range(start, end):
                            if j == start or j == end - 1:
                                stage_str[j] = "FfBbWw"[_cat * 2 + _chunk]
                            elif j == start + 1:
                                if _micro >= 10:
                                    stage_str[j] = str(_micro // 10)
                                else:
                                    stage_str[j] = str(_micro)
                            elif j == start + 2 and _micro >= 10:
                                stage_str[j] = str(_micro % 10)
                            else:
                                stage_str[j] = "-"
            _str = ""
            for _c in stage_str:
                _str += _c
            print(_str)

    def get_v_schedule(self, only_run_time=False):
        schedule, end_time, max_bubble = None, None, None
        expected_time = sum(self.fbw_cost) * self.n_micro * 2
        for fill_b in [True, False]:
            for fill_f in [True, False]:
                _schedule, _end_time, _max_bubble = self.try_v_schedule(fill_b=fill_b, fill_f=fill_f)
                if max_bubble is None or _max_bubble < max_bubble:
                    max_bubble = _max_bubble
                    schedule = _schedule
                    end_time = _end_time
        if only_run_time:
            return max_bubble + expected_time
        max_bubble / (expected_time + max_bubble)
        local_order = [[] for _ in range(self.n_stage)]
        comm_id = {}
        comm_id_counter = 0
        post_validation_time = 0
        for i in range(self.n_stage - 1, -1, -1):
            pv_id = min(2 * (self.n_stage - 1 - i), self.n_micro - 1)
            post_validation_time = max(
                post_validation_time, end_time[self.get_id(0, 0, i, pv_id)] - self.fbw_cost[0] - self.c_cost
            )
            # post_validation_time = 0
            for it in ["RECV_", "SEND_", ""]:
                if i == 0 and it == "SEND_":
                    continue
                if i == self.n_stage - 1 and it == "RECV_":
                    continue
                # stage_ = i - 1 if it == "RECV_" else i
                stage_ = i
                local_order[stage_].append(
                    ScheduledNode(
                        type=it + "POST_VALIDATION",
                        chunk=0,
                        stage=stage_,
                        minibatch=0,
                        start_time=post_validation_time,
                        completion_time=post_validation_time,
                    )
                )
                comm_id[local_order[stage_][-1]] = comm_id_counter
                comm_id_counter += 1
        for i in range(self.n_stage):
            for _cat_, _chunk_, _micro_ in schedule[i]:
                complete_time = end_time[self.get_id(_cat_, _chunk_, i, _micro_)]
                local_order[i].append(
                    ScheduledNode(
                        type="FBW"[_cat_],
                        chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                        stage=i,
                        minibatch=_micro_,
                        start_time=complete_time - self.fbw_cost[_cat_],
                        completion_time=complete_time,
                    )
                )
                if _cat_ == 2:  # no communication for W
                    continue
                cat_str = "FORWARD" if _cat_ == 0 else "BACKWARD"

                def communicate(send_recv, stage_):
                    # noinspection PyTypeChecker
                    local_order[stage_].append(
                        ScheduledNode(
                            type=send_recv + cat_str,
                            chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                            stage=stage_,
                            minibatch=_micro_,
                            start_time=complete_time,
                            completion_time=complete_time,
                        )
                    )
                    comm_id[local_order[stage_][-1]] = comm_id_counter

                if _chunk_ == 1 and i > 0:
                    communicate("SEND_", i)
                    communicate("RECV_", i - 1)
                if _chunk_ == 0 and i < self.n_stage - 1:
                    communicate("SEND_", i)
                    communicate("RECV_", i + 1)
                comm_id_counter += 1
        for rank in range(self.n_stage):
            # For nodes with the same timestamp on the same stage, communication will be prioritized.
            def even_breaker(x: ScheduledNode):
                # Compute nodes are always delayed.
                if x.type in ["F", "B", "W"]:
                    return comm_id_counter
                # For comm nodes, order by their unique comm id
                return comm_id[x]

            local_order[rank] = list(sorted(local_order[rank], key=lambda x: (x.start_time, even_breaker(x))))
            # If a recv with intersects with previous computation, reorder them so that recv
            # is executed before computation and hence can be overlapped.
            for i in range(len(local_order[rank])):
                if (
                    i > 0
                    and local_order[rank][i - 1].type in {"F", "B", "W"}
                    and local_order[rank][i].type.startswith("RECV")
                    and "POST_VALIDATION" not in local_order[rank][i].type
                    and local_order[rank][i].start_time <= local_order[rank][i - 1].completion_time
                ):
                    local_order[rank][i], local_order[rank][i - 1] = local_order[rank][i - 1], local_order[rank][i]

        local_order_with_rollback = [[] for _ in range(self.n_stage)]
        for rank in range(self.n_stage):
            rollback_comm = set()
            if rank > 0:
                for node in local_order[rank - 1]:
                    if node.type == "POST_VALIDATION":
                        break
                    if node.type == "SEND_FORWARD":
                        assert node.chunk == 0
                        rollback_comm.add(node.minibatch)
            for node in local_order[rank]:
                if node.type == "RECV_FORWARD" and node.chunk == 0 and node.minibatch in rollback_comm:
                    rollback = True
                    rollback_comm.remove(node.minibatch)
                else:
                    rollback = False
                local_order_with_rollback[rank].append(
                    ScheduledNode(
                        type=node.type,
                        chunk=node.chunk,
                        stage=node.stage,
                        minibatch=node.minibatch,
                        start_time=node.start_time,
                        completion_time=node.completion_time,
                        rollback=rollback,
                    )
                )
            assert len(rollback_comm) == 0

        return local_order_with_rollback


# DualV integer programming version
class DualVPipelineGraph_IP(PipelineGraph):
    """DualVPipelineGraph: A cut-in-half combination of DualPipe and Zerobubble V"""

    def __init__(
        self,
        n_stage,
        n_micro,
        f_cost,
        b_cost,
        w_cost,
        c_cost,
        f_mem,
        b_mem,
        w_mem,
        max_mem=None,
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

    def convert_to_dualV(self, pipeline_schedule: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
        """
        convert zbv to dualV, spec convert parital B&W to Fully Backward. To save memory for caching dx
        """
        dualV_schedules = [[] for _ in range(self.n_stage)]
        for stage in range(self.n_stage):
            for node in pipeline_schedule[stage]:
                if node.type == "B":
                    if node.chunk == 1 and node.minibatch in range(self.n_stage - 1 - stage, self.n_micro - 1 - stage):
                        dualV_schedules[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=node.chunk,
                                stage=node.stage,
                                minibatch=node.minibatch,
                                start_time=node.start_time,
                                completion_time=node.completion_time,
                            )
                        )
                    elif node.chunk == 0 and node.minibatch in range(
                        self.n_micro - self.n_stage - self.n_stage + stage
                    ):
                        dualV_schedules[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=node.chunk,
                                stage=node.stage,
                                minibatch=node.minibatch,
                                start_time=node.start_time,
                                completion_time=node.completion_time,
                            )
                        )
                    else:
                        dualV_schedules[stage].append(node)
                elif node.type == "W":
                    if node.chunk == 1 and node.minibatch in range(self.n_stage - 1 - stage, self.n_micro - 1 - stage):
                        pass
                    elif node.chunk == 0 and node.minibatch in range(
                        self.n_micro - self.n_stage - self.n_stage + stage
                    ):
                        pass
                    else:
                        dualV_schedules[stage].append(node)
                else:
                    dualV_schedules[stage].append(node)

        return dualV_schedules
    
    def overlap_optim(self, schedules: List[List[ScheduledNode]]):
        def optimize_last_stage_schedule(schedules: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
            """
            优化流水线并行中最后阶段的调度，以实现最大可能的计算重叠
            
            参数:
            schedules - 每个阶段的调度列表，每个调度是一个ScheduledNode列表
            
            返回:
            优化后的调度列表
            """
            if not schedules:
                return []
            
            # 复制原始调度，避免修改
            optimized_schedules = copy.deepcopy(schedules)
            
            # 获取最后阶段的调度
            last_stage = len(schedules) - 1
            last_stage_schedule = optimized_schedules[last_stage]
            
            # 分离前向传播(F)、输入求导(B)、权重求导(W)和Full_B操作
            f_nodes = []
            b_nodes = []
            w_nodes = []
            full_b_nodes = []
            
            for node in last_stage_schedule:
                if node.type == 'F':
                    f_nodes.append(node)
                elif node.type == 'B':
                    b_nodes.append(node)
                elif node.type == 'W':
                    w_nodes.append(node)
                elif node.type == 'Full_B':
                    full_b_nodes.append(node)
            
            # 优化前向传播操作的顺序以实现最大重叠
            optimized_f_nodes = optimize_f_nodes(f_nodes)
            
            # 优化反向传播操作的顺序
            optimized_full_b_nodes = sort_by_minibatch(full_b_nodes)
            optimized_b_nodes = sort_by_minibatch(b_nodes)
            optimized_w_nodes = sort_by_minibatch(w_nodes)
            
            # 重新组合优化后的调度
            optimized_last_stage_schedule = (
                optimized_f_nodes + 
                optimized_full_b_nodes + 
                merge_b_and_w_nodes(optimized_b_nodes, optimized_w_nodes)
            )
            
            # 更新优化后的调度
            optimized_schedules[last_stage] = optimized_last_stage_schedule
            
            # 重新计算时间戳（简化版本，实际应用中可能需要更复杂的时间计算）
            recalculate_timestamps(optimized_schedules)
            
            return optimized_schedules

        def optimize_f_nodes(f_nodes: List[ScheduledNode]) -> List[ScheduledNode]:
            """优化前向传播节点的顺序以实现最大重叠"""
            if not f_nodes:
                return []
            
            # 按微批次ID排序，然后按阶段排序
            sorted_f_nodes = sorted(f_nodes, key=lambda x: (x.minibatch, x.stage))
            
            # 使用贪心算法尝试找到最佳重叠顺序
            # 这里简化为交替排列不同微批次的F操作
            minibatches = sorted({node.minibatch for node in sorted_f_nodes})
            batches_by_minibatch = {mb: [] for mb in minibatches}
            
            # 按微批次分组
            for node in sorted_f_nodes:
                batches_by_minibatch[node.minibatch].append(node)
            
            # 按轮询方式组合不同微批次的F操作
            optimized = []
            max_len = max(len(batches) for batches in batches_by_minibatch.values())
            
            for i in range(max_len):
                for mb in minibatches:
                    if i < len(batches_by_minibatch[mb]):
                        optimized.append(batches_by_minibatch[mb][i])
            
            return optimized

        def sort_by_minibatch(nodes: List[ScheduledNode]) -> List[ScheduledNode]:
            """按微批次ID排序节点"""
            return sorted(nodes, key=lambda x: x.minibatch)

        def merge_b_and_w_nodes(b_nodes: List[ScheduledNode], w_nodes: List[ScheduledNode]) -> List[ScheduledNode]:
            """合并B和W节点，尽量使它们连续执行"""
            merged = []
            
            # 创建微批次到B和W节点的映射
            b_by_minibatch = {}
            for node in b_nodes:
                if node.minibatch not in b_by_minibatch:
                    b_by_minibatch[node.minibatch] = []
                b_by_minibatch[node.minibatch].append(node)
            
            w_by_minibatch = {}
            for node in w_nodes:
                if node.minibatch not in w_by_minibatch:
                    w_by_minibatch[node.minibatch] = []
                w_by_minibatch[node.minibatch].append(node)
            
            # 合并B和W节点
            all_minibatches = sorted(set(b_by_minibatch.keys()).union(set(w_by_minibatch.keys())))
            
            for mb in all_minibatches:
                if mb in b_by_minibatch:
                    merged.extend(b_by_minibatch[mb])
                if mb in w_by_minibatch:
                    merged.extend(w_by_minibatch[mb])
            
            return merged

        def recalculate_timestamps(schedules: List[List[ScheduledNode]]) -> None:
            """重新计算所有节点的时间戳"""
            # 简化实现，实际应用中可能需要考虑操作间的依赖关系和执行时间
            current_time = 0
            for stage in range(len(schedules)):
                for i in range(len(schedules[stage])):
                    node = schedules[stage][i]
                    new_node = ScheduledNode(
                        type=node.type,
                        chunk=node.chunk,
                        stage=node.stage,
                        minibatch=node.minibatch,
                        start_time=current_time,
                        completion_time=current_time + 1,
                    )
                    current_time += 1
                    schedules[stage][i] = new_node
                # for node in schedules[stage]:
                #     node.start_time = current_time
                #     # 假设每个操作耗时1个单位
                #     node.completion_time = current_time + 1
                #     current_time += 1
        
        return optimize_last_stage_schedule(schedules)
    

# Reference From https://github.com/deepseek-ai/DualPipe/blob/main/dualpipe/dualpipev.py
class DualVPipelineGraph(object):
    def __init__(
        self,
        pp_size: int = 1, 
        num_microbatch: int = 1, 
    ) -> None:
        self.pp_size = pp_size
        self.num_microbatch = num_microbatch
        self.dualV_schedules: List[List[VScheduledNode]] = [[] for _ in range(pp_size)]

        self.current_f_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_b_chunk_id: List[List[int]]= [[0, 0] for _ in range(self.pp_size)]
        self.current_w_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_send_f_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_send_b_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_recv_f_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        self.current_recv_b_chunk_id: List[List[int]] = [[0, 0] for _ in range(self.pp_size)]
        
    def forward_compute_schedule(self, phase: int, stage:int, microbatch_id: int) -> None:
        self.dualV_schedules[stage].append(
            VScheduledNode(
                type=FuncType.F,
                chunk=phase,
                stage=stage,
                minibatch=microbatch_id,
                start_time=0,
                completion_time=0, 
            )
        )
        
        # self.current_f_chunk_id[phase] += 1

    def backward_compute_schedule(self, phase: int, stage:int, microbatch_id: int, enable_zb: bool = False) -> None:
        # chunk_id = self.current_b_chunk_id[phase]
        self.dualV_schedules[stage].append(
            VScheduledNode(
                type=FuncType.B if enable_zb else FuncType.BW,
                chunk=phase,
                stage=stage,
                minibatch=microbatch_id,
                start_time=0,
                completion_time=0, 
            )
        )
        # self.current_b_chunk_id[phase] += 1

    def forward_backward_compute_schedule(self, phase0: int, phase1: int, stage:int, fwd_microbatch_id: int, bwd_microbatch_id: int, ) -> None:
        self.forward_compute_schedule(phase0, stage, fwd_microbatch_id)
        self.backward_compute_schedule(phase1, stage, bwd_microbatch_id)
        return

    def forward_schedule(self, phase: int, stage: int, microbatch_id: int, recv: bool = True, send: bool = True) -> None:
        # phase: model chunk id , 0 is left chunk, 1 is right chunk
        if recv:
            self.recv_forward_schedule(phase, stage, microbatch_id)

        self.forward_compute_schedule(phase, stage, microbatch_id)

        if send:
            self.send_forward_schedule(phase, stage, microbatch_id)

    def backward_schedule(self, phase: int, stage:int, microbatch_id: int, enable_zb: bool = False, recv: bool = True, send: bool = True) -> None:
        if recv:
            self.recv_backward_schedule(phase, stage, microbatch_id)

        self.backward_compute_schedule(phase, stage, microbatch_id, enable_zb=enable_zb)

        if send:
            self.send_backward_schedule(phase, stage, microbatch_id)

    def forward_backward_schedule(self, phase0: int, phase1: int, stage:int, fwd_microbatch_id: int, bwd_microbatch_id: int, recv0: bool = True) -> None:
        if recv0:
            self.recv_forward_schedule(phase0, stage, fwd_microbatch_id,)
        self.recv_backward_schedule(phase1, stage, bwd_microbatch_id,)

        self.forward_backward_compute_schedule(phase0, phase1, stage, fwd_microbatch_id, bwd_microbatch_id)

        self.send_forward_schedule(phase0, stage, fwd_microbatch_id,)
        self.send_backward_schedule(phase1, stage, bwd_microbatch_id,)

    def weight_compute_schedule(self, phase: int, stage:int, microbatch_id: int, ) -> None:
        self.dualV_schedules[stage].append(
            VScheduledNode(
                type=FuncType.W,
                chunk=phase,
                stage=stage,
                minibatch=microbatch_id,
                start_time=0,
                completion_time=0, 
            )
        )

    def recv_forward_schedule(self, phase: int, stage:int, microbatch_id: int) -> None:
        # Append recv forward schedule Node to schedule[rank]
        
        # first stage and phase 0 no recv
        # last stage and phase 1 no recv
        if (stage == 0 and phase == 0) or (stage == (self.pp_size - 1) and phase == 1):
            return
        self.dualV_schedules[stage].append(
            VScheduledNode(
                type=FuncType.RECV_FORWARD,
                chunk=phase,
                stage=stage,
                minibatch=microbatch_id,
                start_time=0,
                completion_time=0, 
            )
        )
        # self.current_recv_f_chunk_id[phase] += 1
        return 

    def send_forward_schedule(self, phase: int, stage:int, microbatch_id: int) -> None:
        if (stage == 0 and phase == 1) or (stage == (self.pp_size - 1) and phase == 0):
            return
        self.dualV_schedules[stage].append(
            VScheduledNode(
                type=FuncType.SEND_FORWARD,
                chunk=phase,
                stage=stage,
                minibatch=microbatch_id,
                start_time=0,
                completion_time=0, 
            )
        )
        # self.current_send_f_chunk_id[phase] += 1

    def recv_backward_schedule(self, phase: int, stage:int, microbatch_id: int) -> None:
        if (stage == 0 and phase == 1) or (stage == (self.pp_size - 1) and phase == 0):
            return
        self.dualV_schedules[stage].append(
            VScheduledNode(
                type=FuncType.RECV_BACKWARD,
                chunk=phase,
                stage=stage,
                minibatch=microbatch_id,
                start_time=0,
                completion_time=0, 
            )
        )
        # self.current_recv_b_chunk_id[phase] += 1
        return

    def send_backward_schedule(self, phase: int, stage:int, microbatch_id: int) -> None:
        if (stage == 0 and phase == 0) or (stage == (self.pp_size - 1) and phase == 1):
            return
        self.dualV_schedules[stage].append(
            VScheduledNode(
                type=FuncType.SEND_BACKWARD,
                chunk=phase,
                stage=stage,
                minibatch=microbatch_id,
                start_time=0,
                completion_time=0, 
            )
        )
        # self.current_send_b_chunk_id[phase] += 1

    def get_dual_v_schedule(
        self, 
        additional_warmup: int = 0,
    ) -> List[List[ScheduledNode]]:
        """
        """
        
        for rank in range(self.pp_size):
            # Step 1: nF0
            step_1 = (self.pp_size - rank - 1) * 2
            for i in range(step_1):
                self.forward_schedule(phase=0,stage=rank, microbatch_id=i)

            # Step 2: nF0F1
            step_2 = rank + 1 + additional_warmup
            self.recv_forward_schedule(phase=0, stage=rank, microbatch_id=step_1 + 1)
            for i in range(step_2):
                self.forward_schedule(phase=0, stage=rank, microbatch_id=step_1 + i, recv=False, send=False)
                self.recv_forward_schedule(phase=0, stage=rank, microbatch_id=step_1 + 1)
                self.forward_schedule(phase=1, stage=rank, microbatch_id=i, send=(not rank==(self.pp_size-1)) or (i < step_2 - 1))
                self.send_forward_schedule(phase=0, stage=rank, microbatch_id=step_1 + 1)

            # Step 3: nB1W1F1 (Use zero bubble)
            step_3 = self.pp_size - rank - 1 + additional_warmup
            for i in range(step_3):
                self.backward_schedule(phase=1, stage=rank, microbatch_id=i, enable_zb=True)
                self.recv_forward_schedule(phase=1, stage=rank, microbatch_id=i + rank + 1,)
                self.weight_compute_schedule(phase=1, stage=rank, microbatch_id=i,)
                self.forward_schedule(phase=1,stage=rank, microbatch_id=i + rank + 1, recv=False)

            # Step 4 (Main step): nF0B1F1B0
            step_4 = self.num_microbatch - self.pp_size * 2 + rank + 1 + additional_warmup
            for i in range(step_4):
                if i == 0:
                    if rank == (self.pp_size - 1):
                        # NOTE: We don't overlap these two chunks to further reduce bubble size.
                        self.forward_schedule(phase=0, stage=rank, microbatch_id=step_1+step_2,recv=False, send=False)
                        self.send_forward_schedule(phase=1, stage=rank, microbatch_id=step_1+step_2,)
                        self.backward_schedule(phase=1, stage=rank, microbatch_id=step_3, send=False)
                        self.send_forward_schedule(phase=0, stage=rank, microbatch_id=step_1+step_2,)
                        self.send_backward_schedule(phase=1, stage=rank, microbatch_id=step_3,)
                    else:
                        self.forward_backward_schedule(phase0=0, phase1=1, stage=rank, fwd_microbatch_id=step_1+step_2,bwd_microbatch_id=step_3, recv0=False)
                else:
                    self.forward_backward_schedule(phase0=0, phase1=1, stage=rank, fwd_microbatch_id=step_1+step_2+i,bwd_microbatch_id=self.pp_size - rank + i - 1)
                self.forward_backward_schedule(phase0=0, phase1=1, stage=rank, fwd_microbatch_id= step_4 - rank + i + 1,bwd_microbatch_id=i)

            # Step 5: nB1F1B0
            step_5 = self.pp_size - rank - 1 + additional_warmup
            for i in range(step_5):
                self.backward_schedule(phase=1, stage=rank, microbatch_id=step_4 + step_5 + i)
                self.forward_backward_schedule(phase0=1, phase1=0, stage=rank, fwd_microbatch_id=step_2 + step_4 + step_5 + i ,bwd_microbatch_id=step_4 + i)
            
        
            # Step 6: nB1B0 (The second half of the chunks use zero bubble)
            step_6 = rank + 1 + additional_warmup
            enable_zb = False
            for i in range(step_6):
                if i == step_6 // 2 and rank % 2 == 1:
                    enable_zb = True
                self.backward_schedule(phase=1, stage=rank, microbatch_id=step_3 + step_4 + step_5 + i, enable_zb=enable_zb)
                if i == step_6 // 2 and rank % 2 == 0:
                    enable_zb = True
                self.backward_schedule(phase=0, stage=rank, microbatch_id=step_3 + step_4 + i, enable_zb=enable_zb)

            # Step 7: nWB0 (Use zero bubble)
            step_7 = self.pp_size - rank - 1 + additional_warmup
            # find last Full_Backward(BW) micobatch id phase 0 / 1
            
            phase0_last_BW_microbatch_id = 0
            phase1_last_BW_microbatch_id = 0
            for node in self.dualV_schedules[rank]:
                if node.type == FuncType.BW:
                    if node.chunk == 0:
                        phase0_last_BW_microbatch_id = node.minibatch
                    elif node.chunk == 1:
                        phase1_last_BW_microbatch_id = node.minibatch
            phase0_rest_W = [i for i in range(phase0_last_BW_microbatch_id + 1, self.num_microbatch + additional_warmup)]
            phase1_rest_W = [i for i in range(phase1_last_BW_microbatch_id + 1, self.num_microbatch + additional_warmup)]
            for i in range(step_7):
                microbatch_id = self.num_microbatch - 1
                phase = 0
                if phase1_rest_W:
                    phase = 1
                    microbatch_id = phase1_rest_W.pop(0)
                else:
                    if phase0_rest_W:
                        phase = 0
                        microbatch_id = phase0_rest_W.pop(0)
                
                self.weight_compute_schedule(phase=phase, stage=rank, microbatch_id=microbatch_id)
                self.backward_schedule(phase=0, stage=rank, microbatch_id=step_4 + step_5 + step_6 + i, enable_zb=True)

            # Step 8: nW
            step_8 = rank + 1 + additional_warmup
            for i in range(step_8):
                microbatch_id = self.num_microbatch - 1
                phase = 0
                if phase1_rest_W:
                    phase = 1
                    microbatch_id = phase1_rest_W.pop(0)
                else:
                    if phase0_rest_W:
                        phase = 0
                        microbatch_id = phase0_rest_W.pop(0)
                self.weight_compute_schedule(phase=0, stage=rank, microbatch_id=microbatch_id)

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
