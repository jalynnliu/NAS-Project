.. NAS-Project documentation master file, created by
   sphinx-quickstart on Fri Jan  3 12:18:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NAS-Project's documentation!
=======================================
NAS (Neural Architecture Search) 算法，一种用于自动搜索比人工设计表现更优异的神经网络的自动机器学习算法。

本项网络结构搜索技术将网络结构视为一个有向无环图（DAG），图中的每一条边表示数据流向，每一个节点表示对数据的操作。在搜索过程中，将搜索分为两部分，第一部分是对DAG图的拓扑结构进行搜索，第二部分是在拓扑结构上对每个节点的操作配置进行优化。并采用竞争的方式将二者进行结合，得到最终的网络结构。

对于拓扑结构的搜索，我们采用枚举的方式得到所有可能的拓扑结构，此时拓扑结构上仅有数据流向，并没有对每个节点赋予操作配置；我们对每个拓扑结构上的操作配置视为一个黑盒优化问题，利用非梯度优化算法进行求解。所以在搜索之前我们需要对拓扑结构空间（网络的深度和宽度）和操作配置空间（每个节点上的候选操作）进行设置。在通过竞赛确定最优网络结构时，需要利用一个评估子程序对每个网络结构进行评估，所以对于具体任务需要定制评估标准和评估程序。

对于整体网络结构搜索，我们采用基于block的搜索模式进行，即对于较深的网络，我们先搜索第一个block，在确定第一个block之后再搜索第二个block，依次类推，完成整个网络结构的搜索。当只设置有一个block的时候就退化为直接对整体网络结构进行搜索。图~\ref{nas-parameters}示意了基于block的网络搜索过程，并给出了在每个部分中的控制参数。

.. image:: parameters.pdf


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Tutorial
   Nas Reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
