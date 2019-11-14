import queue
import sys, os
import multiprocessing
from info_str import (
    NAS_CONFIG,
    CUR_VER_DIR,
    EVALOG_PATH_TEM,
    NETWORK_INFO_PATH,
    WINNER_LOG_PATH,
    LOG_EVAINFO_TEM,
    LOG_EVAFAIL,
    LOG_WINNER_TEM,
    SYS_EVAFAIL,
    SYS_EVA_RESULT_TEM,
    SYS_ELIINFO_TEM,
    SYS_INIT_ING,
    SYS_I_AM_PS,
    SYS_I_AM_WORKER,
    SYS_ENUM_ING,
    SYS_SEARCH_BLOCK_TEM,
    SYS_WORKER_DONE,
    SYS_WAIT_FOR_TASK,
    SYS_CONFIG_ING,
    SYS_GET_WINNER,
    SYS_BEST_AND_SCORE_TEM,
    SYS_START_GAME_TEM,
    SYS_CONFIG_OPS_ING
)


class Communication:
    def __init__(self):
        self.task = queue.Queue()
        self.result = queue.Queue()
        self.idle_gpuq = multiprocessing.Queue()
        for gpu in range(NAS_CONFIG['num_gpu']):
            self.idle_gpuq.put(gpu)


def save_info(path, network, round, original_index, network_num):
    tmpa = 'number of scheme: {}\n'
    tmpb = 'graph_part: {}\n'
    tmpc = '    graph_full: {}\n    cell_list: {}\n    score: {}\n'
    s = LOG_EVAINFO_TEM.format(len(network.pre_block) + 1, round, original_index, network_num)
    s = s + tmpa.format(len(network.item_list))
    s = s + tmpb.format(str(network.graph_part))

    for item in zip(network.graph_full_list, network.cell_list, network.score_list):
        s = s + tmpc.format(str(item[0]), str(item[1]), str(item[2]))

    with open(path, 'a') as f:
        f.write(s)


def list_swap(ls, i, j):
    cpy = ls[i]
    ls[i] = ls[j]
    ls[j] = cpy


def try_secure(func_type, func, args=(), f=None, in_child=False):
    eva_try_again = "\nevaluating failed and we will try again...\n"
    eva_return_direct = "\nevaluating failed many times and we return score 0.05 directly...\n"
    spl_try_again = "\nsample failed and we will try again...\n"
    spl_exit = "\nsample failed many times and we stop the program...\n"
    count = 0

    while True:
        try:
            result = func(*args)
            break
        except Exception as e:
            count += 1
            if func_type == "eva":
                print(e, eva_try_again)
                if f:
                    f.write("\n"+str(e)+eva_try_again)
            if func_type == "spl":
                print(e, spl_try_again)
            if count > 10:
                if func_type == "eva":
                    print(e, eva_return_direct)
                    if f:
                        f.write("\n"+str(e)+eva_return_direct)
                    result = 0.05
                if func_type == "spl":
                    print(e, spl_exit)
                    if in_child:
                        result = None
                    else:
                        sys.exit(0)
                break
    return result
