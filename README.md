# 2026 南京邮电大学校科协 python 组免试题 —— 机器学习系统方向

机器学习与深度学习已经成为了当下最热门的技术之一，机器学习的快速发展和广泛使用很大程度上得益于一系列简单好用且强大的编程框架，例如 Pytorch 和 Tensorflow 等等。但大多数从业者只是这些框架的"调包侠"，对于这些框架内部的细节实现却了解甚少。如果你希望从事深度学习底层框架的开发，或者只是像我一样好奇这些框架的内部实现，那么这道题将会是一个很好的起点。

### 题目要求

本题中你将对 mnist 数据集进行分类任务的实现。不同于一般的机器学习任务，pytorch、scikit-learn 等框架在本题中不被允许使用，你将使用原生 Python（借助 numpy 库）实现一个基础的 softmax 回归算法，以及一个简单的两层神经网络。你还将使用原生 C/C++ 和 cuda 实现 softmax 回归。过程中我们会提供一些关于各个函数实现方式的建议，但具体细节总体上由你自行决定。需要特别说明的是，在原生 python 版本中你可以使用 numpy 的线性代数运算以代替显式使用循环，如果你这么做将是很大的加分项。

---

## 完成情况

| # | 内容 | 文件 | 状态 |
|---|------|------|------|
| 1 | `add()` | `src/simple_ml.py` | ✅ |
| 2 | `parse_mnist()` | `src/simple_ml.py` | ✅ |
| 3 | `softmax_loss()` | `src/simple_ml.py` | ✅ |
| 4 | `softmax_regression_epoch()` | `src/simple_ml.py` | ✅ |
| 5 | `nn_epoch()`（两层神经网络反向传播） | `src/simple_ml.py` | ✅ |
| 6 | `softmax_regression_epoch_cpp()`（原生 C++，手工矩阵乘法） | `src/simple_ml_ext.cpp` | ✅ |
| 7 | `softmax_gradient_kernel` / `update_theta_kernel`（CUDA） | `src/simple_ml_cuda.cu` | ✅ 代码完成，本机无 N 卡未实机运行 |

`pytest` 结果：

```
tests/test_simple_ml.py::test_add                          PASSED
tests/test_simple_ml.py::test_parse_mnist                  PASSED
tests/test_simple_ml.py::test_softmax_loss                 PASSED
tests/test_simple_ml.py::test_softmax_regression_epoch     PASSED
tests/test_simple_ml.py::test_nn_epoch                     PASSED
tests/test_simple_ml.py::test_softmax_regression_epoch_cpp PASSED
tests/test_simple_ml.py::test_softmax_regression_epoch_cuda SKIPPED (无 CUDA GPU)
6 passed, 1 skipped
```

## 目录结构

```
data/                      MNIST 数据集
src/
  simple_ml.py             问题 1-5：numpy 实现
  simple_ml_ext.cpp        问题 6：C++ + pybind11 实现
  simple_ml_cuda.cu        问题 7：CUDA kernel
tests/test_simple_ml.py    课程提供的公开测试
scripts/
  build_windows.sh         Windows/MinGW 下编译 C++ 扩展（见下文）
  verify_cuda_logic.py     在没有 GPU 的机器上离线校验 CUDA kernel 的索引算术
  train_mnist.py           三个训练实验 + 计时，产出 docs/train_mnist_log.txt
docs/train_mnist_log.txt   训练日志
Makefile                   编译规则
```

## 运行方式

依赖：`numpy`、`pytest`、`numdifftools`（数值梯度校验）、`pybind11`（编译 C++ 扩展）。

```bash
pip install numpy pytest numdifftools pybind11
```

**Linux / macOS** —— 按原 Makefile 编译：

```bash
make                        # 生成 src/simple_ml_ext.so
python3 -m pytest -v
python3 src/simple_ml.py    # 训练 softmax 回归 + 两层网络
python3 scripts/train_mnist.py
```

**Windows / MinGW** —— Python 在 Windows 上只认 `.pyd`（不认 `.so`），且 MinGW 的 `ld` 无法直接使用 MSVC 格式的 `libs/python313.lib`，所以用一个脚本处理这两件事：

```bash
PYTHON=/path/to/python.exe bash scripts/build_windows.sh
python -m pytest -v
python scripts/train_mnist.py
```

**CUDA（可选）** —— 需要 Linux + NVIDIA GPU + `nvcc`：

```bash
make cuda
python3 -m pytest -k "softmax_regression_epoch_cuda"
```

没有 GPU 时该测试会自动跳过；本机上我还用 `scripts/verify_cuda_logic.py` 把两个 kernel 的循环结构逐行搬到 CPU 上用 `float32` 复算，与测试里的 numpy 参考实现对拍，最大偏差 `1.1e-8`。

## 训练结果

完整日志见 [`docs/train_mnist_log.txt`](docs/train_mnist_log.txt)。

| 实验 | 超参数 | 测试误差 | 耗时 | 题面参考值 |
|------|--------|----------|------|------------|
| numpy softmax 回归 | 10 epoch, lr=0.2, batch=100 | **7.97%** | 0.61 s | 7.97% |
| C++ softmax 回归 | 同上 | **7.97%** | 4.03 s | 与 python 一致 |
| 两层神经网络 | 400 隐层, 20 epoch, lr=0.2 | **1.95%** | 117.7 s | 1.89% |

两个 softmax 版本的 loss 与误差**逐位相同**（差异 `0.00e+00`），但 C++ 版慢了 **6.62 倍**——正是问题 6 想说明的现象：numpy 的矩阵乘法底层是高度优化的 BLAS（分块、向量化、缓存友好），手写的三重循环比不过它。这恰好是 MLSys 存在的理由。

## 基本设计

### `parse_mnist`

MNIST 用 IDX 二进制格式，大端存储、gzip 压缩。用 `struct.unpack(">IIII", ...)` 读 16 字节头拿到形状，再把剩余 payload 用 `np.frombuffer` 一次性映射成数组，全程没有逐像素的 Python 循环。归一化除以 **255**（整个数据集统一尺度），不是按单张图片归一到 `[0,1]`——测试里第二个断言（`norm(X[:1000])`）专门用来抓这个错误。

### `softmax_loss`

交叉熵展开成 log-sum-exp 形式，一行搞定、零循环：

```
loss = mean_i ( logsumexp(Z[i]) - Z[i, y_i] )
```

减去行最大值只是把 `exp` 的幅值压下来，数学上完全等价，但不这么做在 logits 较大时会溢出。

### `softmax_regression_epoch`

一个 minibatch 上的梯度是

```
dL/dTheta = X_bᵀ (softmax(X_b Theta) − one_hot(y_b)) / b
```

所以一轮 SGD 就是 `Theta -= lr * 上式`。`theta` 用 `-=` 原地更新（题目要求），每个 batch 只有几次 numpy 调用，没有对样本的 Python 循环。

### `nn_epoch`

对 `logits = ReLU(X W1) W2` 做链式法则：

```
Z1 = X W1,  H = ReLU(Z1),  Z2 = H W2
dZ2 = (softmax(Z2) − one_hot(y)) / b
dW2 = Hᵀ dZ2
dZ1 = (dZ2 W2ᵀ) ⊙ 1[Z1 > 0]        # ReLU 的次梯度
dW1 = Xᵀ dZ1
```

关于 ReLU 导数在 0 处的取值，这里用 `Z1 > 0` 作为掩码（0 处取 0）。测试用 `numdifftools` 数值求导校验了完整的反向传播，`W1`/`W2` 的更新量与数值梯度在 `1e-4` 内一致。

### C++ 版本

`X`、`theta` 都是行主序，对应的一维下标是

```
X[i, j]     -> X[i * n + j]
theta[j, c] -> theta[j * k + c]
```

`Z = X_b @ theta` 和 `theta -= lr/b * X_bᵀ @ G` 两个矩阵乘法全部手写三重循环（题目不允许用外部矩阵库）。`batch=50` 是很小的，所以 logits 只有 `batch*k` 个元素，分配一次、循环复用，最后 `delete[]`。

### CUDA 版本

- `softmax_gradient_kernel`：一个线程负责一个样本。内层先对 `k` 个类别各算一遍 `X[example, :] · theta[:, c]`，边算边记行最大值；第二次遍历把 logits 转成 `exp(logit − max)` 并累加归一化因子；第三次遍历除以归一化因子并在真实类别上减 1，得到 `softmax(Z) − one_hot(y)`。分三次遍历是为了数值稳定——必须在知道行最大值之后才能做 `exp`。
- `update_theta_kernel`：一个线程负责 `theta` 的一个元素，`parameter` 直接分解成 `(feature = parameter / k, class_id = parameter % k)`，然后对 minibatch 内所有样本累加 `X[i, feature] * G[i, class_id]`，除以 `batch_size` 后原地写回。

## 遇到的问题与解决

**1. Windows 上编译 C++ 扩展**

原 Makefile 产出 `simple_ml_ext.so`，但 Windows 的 CPython 只识别 `.pyd`，改后缀还不够：

- `ld` 报满屏 `undefined reference to __imp_PyObject_...` —— MinGW 的 `ld` 吃不下 CPython 自带的 MSVC 格式 `libs/python313.lib`。解决：用 `gendef python313.dll` + `dlltool` 从 DLL 现场生成 MinGW 可用的 `libpython313.a`。
- 换过导入库后 `import simple_ml_ext` 报 `DLL load failed: 找不到指定的模块` —— 用 `objdump -p` 看到多了一个 `libwinpthread-1.dll` 依赖，而它不在 Python 的 DLL 搜索路径里。加 `-static-libgcc -static-libstdc++ -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic` 静态链接运行时后解决。

这些步骤都固化在 `scripts/build_windows.sh` 里。

**2. 测试文件 import 不了**

`tests/test_simple_ml.py` 顶部有 `import mugrade`，那是课程的在线提交客户端，本地装不到，导致 pytest 连测试都收集不了。我在本地环境里放了一个 no-op 的 `mugrade` stub（只记录提交内容、不联网），这样 `pytest` 能正常收集运行；仓库本身不含这个文件。

**3. 无 GPU 时怎么保证 CUDA 代码是对的**

`test_softmax_regression_epoch_cuda` 在没有 N 卡时会自动 skip，等于这段代码没被验证过。所以写了 `scripts/verify_cuda_logic.py`：用同样的测试数据（53×7、4 类、batch=16、lr=0.2），把两个 kernel 的循环结构逐行搬到 CPU 上、用 `float32` 标量累加复算，再和测试里的 numpy 参考实现对比。最大偏差 `1.1e-8`，说明下标推导和更新公式是对的。

## 已知限制

- 问题 7（CUDA）在本机（无 NVIDIA GPU、无 `nvcc`）未实机编译运行，仅有上述离线逻辑校验；`make cuda` + 官方测试需要在有 GPU 的 Linux 机器或 Colab 上执行。
- `softmax_regression_epoch` / `nn_epoch` 都按题目要求**不打乱样本顺序**、也不处理最后一个不满的 minibatch 的权重缩放（按实际 batch 大小除，与参考实现一致）。公开测试里所有 batch 都能整除。
- 手写的 C++ 矩阵乘法只做了朴素的三重循环（题目明确不要求优化），因此比 numpy 慢约 6.6 倍；如果要做循环分块 / SIMD，还有很大空间。
