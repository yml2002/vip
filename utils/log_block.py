import os
import warnings
import logging
import tensorflow as tf
import torch
from absl import logging as absl_logging

def log_block():
    # 设置环境变量以屏蔽日志信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['EGL_LOG_LEVEL'] = '2'
    os.environ['CUDNN_LOGINFO_DBG'] = '0'
    os.environ['CUBLAS_LOGINFO_DBG'] = '0'

    # 屏蔽 TensorFlow 的日志信息
    tf.get_logger().setLevel(logging.ERROR)

    # 屏蔽 PyTorch 的日志信息
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('torch.distributed').setLevel(logging.ERROR)

    # 屏蔽 absl 的日志信息
    absl_logging.set_verbosity(absl_logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)

    # 屏蔽 computation_placer 的日志信息
    logging.getLogger('computation_placer').setLevel(logging.ERROR)

    # 屏蔽其他常见日志信息
    logging.getLogger('torch.distributed.run').setLevel(logging.ERROR)
    logging.getLogger('torch.distributed.elastic').setLevel(logging.ERROR)

    # 屏蔽 cuDNN 和 cuBLAS 的日志信息
    logging.getLogger('cuda_dnn').setLevel(logging.ERROR)
    logging.getLogger('cuda_blas').setLevel(logging.ERROR)

    print("++++++++++++++++++++++log_block++++++++++++++++++++++")