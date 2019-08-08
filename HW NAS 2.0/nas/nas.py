import random
import time
import tensorflow as tf
from multiprocessing import Process,Pool

from .base import NetworkUnit
from .enumerater import Enumerater
from .evaluator import Evaluator
from .optimizer import Optimizer
from .sampler import Sampler_table
from .predictor import Predictor

NETWORK_POOL = []


def run_proc(NETWORK_POOL, eva, finetune_signal, first_round, scores):
    for i, nn in enumerate(NETWORK_POOL):
        try:
            if first_round:
                temp_nn = NetworkUnit(nn.graph_full, nn.cell_list[-1])
            else:
                cell, graph = nn.spl.sample()
                nn.graph_full = graph
                nn.cell_list.append(cell)
                temp_nn = NetworkUnit(graph, cell)
            score = eva.evaluate(temp_nn, False, finetune_signal)
            scores.append(score)
            nn.opt.update_model(nn.pros, score)  # ??? here, is the parameter score a list ? or a number ?
            nn.pros = nn.opt.sample()
            nn.spl.renewp(nn.pros)
        except Exception as e:
            print(e)
            return i


class Nas:
    def __init__(self, m_best=1, opt_best_k=5, randseed=-1, depth=6, width=3, max_branch_depth=6):
        self.__m_best = m_best
        self.__m_pool = []
        self.__opt_best_k = opt_best_k
        self.__depth = depth
        self.__width = width
        self.__max_bdepth = max_branch_depth

        if randseed is not -1:
            random.seed(randseed)
            tf.set_random_seed(randseed)

        return

    def __list_swap(self, ls, i, j):
        cpy = ls[i]
        ls[i] = ls[j]
        ls[j] = cpy

    def __eliminate(self, network_pool=None, scores=[]):
        """
        Eliminates the worst 50% networks in network_pool depending on scores.
        """
        scores_cpy = scores.copy()
        scores_cpy.sort()
        mid_val = scores_cpy[len(scores) // 2]

        i = 0
        while (i < len(network_pool)):
            if scores[i] < mid_val:
                # del network_pool[i]   # TOO SLOW !!
                # del scores[i]
                self.__list_swap(network_pool, i, len(network_pool) - 1)
                self.__list_swap(scores, i, len(scores) - 1)
                network_pool.pop()
                scores.pop()
            else:
                i += 1
        return mid_val

    def __datasize_ctrl(self, type="", eva=None):
        """
        Increase the dataset's size in different way
        """
        # TODO Where is Class Dataset?
        cur_train_size = eva.get_train_size()
        if type.lower() == 'same':
            nxt_size = cur_train_size * 2
        else:
            raise Exception("NAS: Invalid datasize ctrl type")

        eva.set_train_size(nxt_size)
        return

    def __save_log(self, path="",
                   optimizer=None,
                   sampler=None,
                   enumerater=None,
                   evaluater=None):
        with open(path, 'w') as file:
            file.write("-------Optimizer-------")
            file.write(optimizer.log)
            file.write("-------Sampler-------")
            file.write(sampler.log)
            file.write("-------Enumerater-------")
            file.write(enumerater.log)
            file.write("-------Evaluater-------")
            file.write(evaluater.log)
        return

    def __run_init(self):
        enu = Enumerater(
            depth=self.__depth,
            width=self.__width,
            max_branch_depth=self.__max_bdepth)
        eva = Evaluator()
        pred = Predictor()

        return enu, eva, pred


    def __game(self, eva, finetune_signal,first_round, NETWORK_POOL):
        print("NAS: Now we have {0} networks. Start game!".format(len(NETWORK_POOL)))
        scores = []
        eva.add_data(800)
        i = 0

        while i < len(NETWORK_POOL):
            print(i)

            with Pool(1) as p:
                key=p.apply(run_proc, args=(NETWORK_POOL[i:], eva, finetune_signal,first_round, scores))
                i+=key
            # try:
            #     p=Pool(1)
            #     key=p.apply(run_proc, args=(NETWORK_POOL[i:], spl, eva, scores,))
            #     # p = Process(target=run_proc, args=(NETWORK_POOL[i:], spl, eva, scores,))
            #     # key = p.start()
            # except Exception:
            #     print(key)
            #     i += key-1
            #     p.close()


        # for nn in NETWORK_POOL:
        #     spl_list = spl.sample(len(nn.graph_part))
        #     nn.cell_list.append(spl_list)
        #     # with Pool(1) as p:
        #     #     score=p.apply(eva.evaluate,args=(nn,))
        #     #     scores.append(score)
        #     #
        #     score = eva.evaluate(nn)
        #     scores.append(score)

        return scores

    def __train_winner(self, eva, NETWORK_POOL):
        best_nn = NETWORK_POOL[0]
        best_opt_score = 0
        best_cell_i = 0
        eva.add_data(-1)  # why is here -1 ?
        for i in range(self.__opt_best_k):
            cell, graph = best_nn.spl.sample()
            best_nn.graph_full = graph
            best_nn.cell_list.append(cell)
            temp_nn = NetworkUnit(graph, cell)
            opt_score = eva.evaluate(temp_nn, True, True)
            if opt_score > best_opt_score:
                best_opt_score = opt_score
                best_cell_i = i
        print(best_opt_score)
        return best_nn, best_cell_i

    def initialize_ops(self, pred, pattern, NETWORK_POOL):
        for network in NETWORK_POOL:  # initialize the full network by adding the skipping and ops to graph_part
            network.pros = network.opt.sample()
            network.spl.renewp(network.pros)
            cell, graph = network.spl.sample()
            network.graph_full = graph
            network.cell_list.append(pred.predictor(network.pre_block, graph))
            if pattern == "Block":  # NASing based on block mode
                network.cell_list[-1] = self.remove_pooling(network.cell_list[-1])

    def remove_pooling(self, cell):
        for i in range(len(cell)):
            if cell[i][0] == "pooling":
                if i == 0:
                    cell[i] = cell[i+1]  # keep same as the previous one
                    continue
                cell[i] = cell[i-1]  # keep same as the next one
        return cell

    def algorithm(self, pattern):
        """
        Algorithm Main Function
        """
        # Step 0: Initialize
        print("NAS start running...")
        enu, eva, pred = self.__run_init()

        # Step 1: Brute Enumerate all possible network structures and initialize spl and opt for every network
        NETWORK_POOL = enu.enumerate()

        print("NAS: Enumerated all possible networks!")
        # Step 2: Search best structure
        finetune_signal = False
        self.initialize_ops(pred, pattern, NETWORK_POOL)
        scores = self.__game(eva, finetune_signal, True, NETWORK_POOL)
        self.__eliminate(NETWORK_POOL, scores)
        while (len(NETWORK_POOL) > 1):
            # Step 3: Sample, train and evaluate every network
            if len(NETWORK_POOL) < 5:
                finetune_signal = True
            scores = self.__game(eva, finetune_signal, False, NETWORK_POOL)

            # Step 4: Eliminate half structures and increase dataset size
            self.__eliminate(NETWORK_POOL, scores)
            # self.__datasize_ctrl("same", epic)

        print("NAS: We got a WINNER!")
        # Step 5: Global optimize the best network
        best_nn, best_cell_i = self.__train_winner(eva, NETWORK_POOL)

        # self.__save_log("", opt, spl, enu, eva)

        return best_nn, best_cell_i

    def run(self, pattern, block_num=None):

        assert pattern == "Global" or pattern == "Block", "running mode must be chose from 'Global' and 'Block'"

        if pattern == "Global":
            return self.algorithm(pattern)
        else:
            # save the bese_nn and best_cell_i,search next block
            assert block_num, "you must give the number of blocks(block_num) in the Block mode"
            for i in range(block_num):
                block = self.algorithm(pattern)
                block.pre_block.append([block.graph_full, block.cell_list])  # or NetworkUnit.pre_block.append()
            return block.pre_block  # or NetworkUnit.pre_block


if __name__ == '__main__':
    nas = Nas()
    print(nas.run())