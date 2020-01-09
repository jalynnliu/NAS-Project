..

NAS-project教程
=====================

快速上手
-----------
运行环境要求
****************

本工程要求的运行环境如下：

+ Python版本3.6及以上
+ tensorflow-gpu,版本1.13或1.14
+ numpy
+ keras

一个简单的例子
****************

本部分我们先以实现 ``cifar-10分类任务`` 的NAS工程为例，说明如何启动整个NAS工程。

1. 定义任务
````````````````

(1). 读入数据
^^^^^^^^^^^^^^
在 `evaluator.py <../../../evaluator.py>`__ 中实现cifar-10的读入和处理。

1). 给定任务数据的描述

定义 ``Evaluator.__init__`` 函数中的参数

.. code-block:: python

 self.batch_size = 50
 self.input_shape = [self.batch_size, image_size, image_size, 3]
 self.output_shape = [self.batch_size, num_class]

2). 完成数据读入

在 ``Dataset`` 类中实现 ``input()`` 方法，返回cifar-10的训练和测试数据集 ``self.train_data`` ,
``self.train_label`` , ``self.test_data`` , ``self.test_label`` ，返回类型为list。

(2). 定义Evaluator._eval函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
利用本函数的给定输入进行计算，返回规定输出。

输入参数：

    + ``sess``：Tensorflow Session
    + ``logits``：Tensorflow Tensor; 网络模型的最后一层输出，注意可能需要进一步处理。
    + ``data_x``：Tensorflow Placeholder; 网络的输入
    + ``data_y``：Tensorflow Placeholder; 真实标签

1). 利用logits计算具体结果

logits为搜索网络得到的输出，为了得到最终结果需要将其经过一层全连接层，将维度压缩至输出大小

.. code-block:: python

  logits = self._makedense(logits, ('', [self.output_shape[-1]], ''))

2). 定义loss的计算方式、梯度的更新方式

定义loss，建议将计算loss的方法封装为一个函数

.. code-block:: Python

   def _loss(self, labels, logits):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = cross_entropy + l2 * self.weight_decay
        return loss

定义train_op：

.. code-block:: Python

    def _train_op(self, global_step, loss):
        num_batches_per_epoch = self.train_num / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.epoch)
        lr = tf.train.cosine_decay(self.INITIAL_LEARNING_RATE, global_step, decay_steps, 0.0001)

        opt = tf.train.MomentumOptimizer(lr, self.momentum_rate, name='Momentum' + str(self.block_num),
                                         use_nesterov=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)
        return train_op

**注意** 这里一定要加上tf.GraphKeys.UPDATE_OPS的依赖，具体原因可见 ``实现自定义任务`` 章节下的 `(2) 定义loss的计算方式、梯度的更新方式`_ 。

3). 定义返回的目标值

.. code:: python

  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

4). 启动 ``sess.run()``

.. code-block:: Python

        log = ''

        for ep in range(self.epoch):
            precision = 0
            for step in range(self.train_num // self.batch_size):
                batch_x = self.train_data[step * self.batch_size:(step + 1) * self.batch_size]
                batch_y = self.train_label[step * self.batch_size:(step + 1) * self.batch_size]
                batch_x = DataSet().process(batch_x)
                _, loss_value, acc = sess.run([train_op, loss, accuracy],
                                              feed_dict={data_x: batch_x, data_y: batch_y, train_flag: True})

            for step in range(10000 // self.batch_size):
                batch_x = test_data[step *
                                    self.batch_size:(step + 1) * self.batch_size]
                batch_y = test_label[step *
                                     self.batch_size:(step + 1) * self.batch_size]
                l, acc = sess.run([loss, accuracy],
                                   feed_dict={data_x: batch_x, data_y: batch_y, train_flag: False})
                precision += acc / num_iter

            log += 'epoch %d: precision = %.3f\n' % (ep, precision)

        return precision, log

``acc`` 、 ``loss_value`` 即为训练过程中的准确率和loss。

**注意** 这里的self.epoch和self.train_num是不可自定义的，且必须使用这两个值。

2. 定义搜索空间
``````````````````
搜索空间包括 ``拓扑结构`` 和 ``操作算子`` 两个。

(1). 定义拓扑结构
^^^^^^^^^^^^^^^^^^^^^
此处设定搜索网络block的深度和宽度。假设想要搜索深度为5、宽度为3的block，且支链上的节点不超过2，只需在 `nas_config.json <../../../nas_config.json>`__
中修改：

.. code-block:: Python

  "enum":{
    "depth": 5,
    "width": 3,
    "max_depth": 2
    }

还可以在下方设置最大跨层长度和最大跨层个数：

.. code-block:: Python

    "spl":{
    "skip_max_dist": 4,
    "skip_max_num": 3
    }

(2). 定义操作算子
^^^^^^^^^^^^^^^^^^^^
注意这个部分可选，因为我们已经实现了部分操作，详见 `2. 修改操作配置`_ .

假设现在要加入卷积（Convolution）操作到神经网络，其中包含filter\_size、kernel\_size和激活函数类型几个参数。
设定几个参数的取值范围为：

+ filter\_size: 32, 48, 64
+ kernel\_size: 1, 3, 5
+ 激活函数类型：relu, leakyrelu, relu6

(1). 首先在 ``nas_config.json/spl/space`` 下按如下格式添加卷积操作的搜索空间：

.. code-block:: Python

  "conv": {
    "filter_size": [   32, 48, 64   ],
    "kernel_size": [     1,     3,     5   ],
    "activation": ["relu",     "leakyrelu",     "relu6"]
  }

(2). 参考 `evaluator.py <../../../evaluator.py>`__ ，在 ``Evaluator._make_layer`` 中的 ``elif`` 下添加新的操作类型，格式如下

.. code-block:: Python

  elif cell.type == 'conv':
       layer = self._makeconv(inputs, cell, node)

然后定义具体操作对应的函数

.. code-block:: Python

                def _makeconv(self, inputs, cell, node):
                    ...
                    return conv_layer

具体示范参见 `evaluator.py <../../../evaluator.py>`__ 中 ``_makeconv`` 函数的具体实现。

3. 启动搜索算法
``````````````````
(1). 运行

::

$ python nas.py

或在新建python文件中输入

.. code-block:: Python

            from multiprocessing import Pool
            from info_str import NAS_CONFIG
            from nas import NAS

            if __name__ == '__main__':
                NUM_GPU = NAS_CONFIG['nas_main']["num_gpu"]
                p = Pool(processes=NUM_GPU, maxtaskperchild=1)
                nas = Nas(p)
                best_nn = nas.run()
                p.close()
                p.join()

``best_nn`` 即为最佳网络

(2). 运行过程中的评估信息、中间结果等日志保存在 ``memory`` 文件夹下。详见 `运行日志 <log.html>`__ .

代码文件夹中附带的 `evaluator_classification.py <../../../evaluator_classification.py>`__ 实现了cifar-10的分类任务，
需要运行的话只需将文件名重命名为evaluator.py，然后按照步骤(5)启动搜索算法即可。如需定义更为复杂的任务，更多详细内容请
参照 `实现自定义任务`_ 以及 `改变搜索空间`_ 中的内容。

如何实现自定义搜索
---------------------
实现自定义任务
****************
这一步需要修改的函数集中在 `evaluator.py <../../../evaluator.py>`__ 中。在完成本步骤之前evaluator.py无法运行。
实现自定义的任务，需要以下几个步骤：

1. 任务数据的读入与处理。
``````````````````````````
(1) 给定任务数据的描述
^^^^^^^^^^^^^^^^^^^^^^^^^^^
这一步需要修改 ``Evaluator.__init__`` 函数中的参数

+ ``self.input_shape`` : list, list中为int; 输入数据尺寸，一般情况下为[batch\_size, H, W, C]，其中H，W，C为图片尺寸
+ ``self.output_shape`` : list, list中为int; 输出数据尺寸。

注意 ``Evaluator.__init__`` 函数中其他的已有参数不需要修改。如果有训练任务中需要的其他参数也可以在这里定义。

(2) 完成数据的读入与处理
^^^^^^^^^^^^^^^^^^^^^^^^^^

数据在 ``Evaluator.__init__`` 函数中的 ``self.train_data`` , ``self.train_label`` ,
``self.test_data`` , ``self.test_label`` 中，
这几个参数的返回类型建议为 ``list`` ，也可以根据任务稍作修改。

**注意数据必须在Evaluator类初始化时读入** ，防止在后期枚举网络时重复读入数据耗时大。
建议将跟数据相关的函数封装为一个类， `evaluator_classification.py <../../../evaluator_classification.py>`__ 中提供了一个示范。

2. 修改 ``Evaluator._eval`` 函数
`````````````````````````````````````

由于具体的任务对应的评估训练方式可能不同，这里具体的评估方式也需自行实现。具体只需要修改 ``Evaluator._eval`` 函数。
本函数的输入和输出已经固定。

- ``Evaluator._eval``

   输入参数：

    + ``sess``：Tensorflow Session; 其中网络构图已经载入Tensorflow，需要利用 ``sess.run()`` 启动训练过程
    + ``logits``：Tensorflow Tensor; 网络模型的预测输出，注意可能需要进一步处理。
    + ``data_x``：Tensorflow Placeholder; 网络的样本（或称特征）输入，可以通过feet\_dict的方式将数据输入计算图中，如果对tensorflow的机制不熟悉需要注意 ``data_x`` 与 ``self.train_data`` 的区别和联系。
    + ``data_y``：Tensorflow Placeholder; 网络的真实标签，可以通过feet\_dict的方式将数据输入计算图中

   输出参数：

    + ``target``：float；网络的评估值。可以是单纯的准确率，或者多目标综合的结果（如综合考虑准确率和模型大小）
    + ``log``：string；可为空，需要打印的日志信息

本函数主要利用 ``sess.run()`` 启动训练过程返回评估值。

(1) 利用logits计算具体结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
这里需要注意logits不一定表示最终输出，只是表示搜索的网络得出的输出。

比如在 ``分类问题`` 中，logits为卷积和池化等操作的输出，还需经过一层全连接将logits由原来的四维压缩为两维，即[batch\_size,num\_class]大小。
详见 `evaluator_classification.py <../../../evaluator_classification.py>`__ 的 ``_eval`` 函数的示例。

再比如在 ``图片去噪`` 问题中，最后需要将图片还原成原来的尺寸，需要在最后一层加一个输出channel为3的卷积层，都需要在这里添加。

(2) 定义loss的计算方式、梯度的更新方式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在

.. code:: python

  loss = self._loss(logits, data_y)

中定义 ``loss`` 的计算。

在

.. code:: python

 train_op = self._train_op(global_step, loss)

中定义梯度的更新方式，返回 ``train_op`` 。这里需要注意由于搜索的算子中有 ``batch_norm`` 的计算， ``train_op``
的计算需要添加依赖，即

.. code:: python

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)

具体原理可参见 `此处 <http://www.jianshu.com/p/437fb1a5823e>`__ 。

(3) 定义返回的目标值
^^^^^^^^^^^^^^^^^^^^^^^^^
在

.. code:: python

  accuracy = self._cal_accuracy(logits, data_y)

中定义任务的目标。

(4) 启动 ``sess.run()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
我们的数据输入方式是定义 ``tf.placeholder`` 后通过feet_dict的方式输入，即

.. code:: python

  acc, loss_value, _ = sess.run(accuracy, loss, train_op, feet_dict={data_x: self.train_data[batch_size], data_y: self.train_label[batch_size]})

其中 ``acc`` 、 ``loss_value`` 分别为训练过程中的准确率和loss。

**注意** 训练中的epoch数不可自定义，需使用self.epoch。而每一个epoch中训练的循环（iteration）不可使用固定数据集
大小，需使用self.train_num。举例来说，cifar-10数据集的训练集大小为50000，则每一个epoch应循环50000/batch_size次，
但这里不可使用50000，而应使用self.train_num，也就是训练应循环self.train_num/batch_size次。
`(2). 修改Evaluator._eval函数`_ 提供了一个示例。

改变搜索空间
********************
算法原理中概述了我们的搜索空间分为两个部分：拓扑结构和操作配置。其中拓扑结构即为网络拓扑结构，操作配置即为基本算子。

1. 修改拓扑结构
``````````````````````

拓扑结构的修改主要是网络的深度和宽度，本部分修改的内容集中在 `nas_config.json <../../../nas_config.json>`__
中，仅需修改几个数字即可控制网络拓扑结构的范围。

- enum 穷举模块参数
    + depth 枚举的网络结构的深度
    + width 枚举的网络结构的支链个数
    + max\_depth 约束支链上节点的最大个数

- spl 采样参数
    + skip\_max\_dist 最大跨层长度
    + skip\_max\_num 最大跨层个数

如果对这部分有疑问可以参见 `此页 <project.html>`__ 的图.

2. 修改操作配置
``````````````````````
注意这个部分是可选的，因为我们已经实现了一部分操作，包括：卷积操作（ ``tf.nn.conv2d`` ）、分离卷积（ ``tf.nn.separable_conv2d`` ）、
池化操作（包括最大池化 ``tf.nn.max_pool`` 、平均池化 ``tf.nn.avg_pool`` 和全局池化 ``global_pooling`` ），其中卷积操作和分离卷积
都包含 ``batch_norm`` 操作。如果不需要添加新的操作则不需要对这个部分做改动即可运行。

本部分修改的内容集中在 `nas_config.json <../../../nas_config.json>`__ 和 `evaluator.py <../../../evaluator.py>`__ 中。

(1). 修改 `nas_config.json <../../../nas_config.json>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
修改nas\_config.json/spl/space，在其下按如下格式添加新操作的搜索空间：

  .. code:: python

    "operation_name":{
    "param1": [   value1, value2...   ],
    "param2": [     value1,  value2...  ],
    ...
    }

(2). 修改 `evaluator.py <../../../evaluator.py>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
在Evaluator.\_make\_layer中的elif下添加新的操作类型，格式如下

.. code:: python

                elif cell.type == 'operation_name':
                    layer = self._name_your_function_here(inputs, cell, node)

然后定义具体操作对应的函数

.. code:: python

                def _name_your_function_here(self, inputs, cell, node):
                    # TODO add your function here if any new operation was added, see _makeconv as an example
                    return layer

此函数命名需要自己填写，但其输出必须为Tensorflow Tensor类型。输入参数含义如下：

+ ``inputs`` ：Tensorflow Tensor；输入
+ ``cell`` ：Cell类；包含操作配置信息
+ ``node`` ：int；当前操作的节点编号，用于对操作节点的op进行唯一命名


其他可以修改的参数
***********************
在nas\_config.json/nas\_main中还有其他可以修改的参数，用以适用具体的任务或与运行环境相匹配。下面介绍一下这些参数以及其具体含义和作用。

+ num\_gpu 运行环境GPU个数
+ block\_num 堆叠网络块数量，详细可见 `此页 <project.html>`__ 图中的编号4
+ repeat\_search 模块重复次数，详细可见 `此页 <project.html>`__ 图中的编号5
+ link\_node 连接节点类型，详细可见 `此页 <project.html>`__ 图中的编号6
+ add\_data\_per\_round 每一轮竞赛增加数据大小
+ add\_data\_for\_winner 竞赛胜利者的训练数据集大小(-1代表使用全部数据)

