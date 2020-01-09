.. NAS-Project documentation master file, created by
   sphinx-quickstart on Fri Jan  3 12:18:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

运行日志
===========

为了方便查看中间结果以及搜索过程，NAS工程中内置了Logger方法用于记录运行中的日志。所有日志文件都放在工程文件夹memory中：

+ 评估过程 evaluator\_log.txt
+ 子进程 subproc\_log.txt
+ 网络信息 network\_info.txt
+ 总控 nas\_log.txt

日志纪录对象Logger提供全工程统一的日志纪录接口，如果需要纪录更多更复杂的日志，可以重新实现Logger。
具体使用说明，请参考doc文件夹下的interface.md。