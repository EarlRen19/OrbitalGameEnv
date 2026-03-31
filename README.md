# OGE

## 1. 开发环境配置

### 1.1 更新 gcc

安装 GCC 13

```
sudo apt update
sudo apt install build-essential gcc-13 g++-13
```

设置为默认版本

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100
```

如果系统里有多个版本，可以用以下命令切换：

```
sudo update-alternatives --config gcc                                                                                                                                                                                
sudo update-alternatives --config g++
```

安装完后验证：

```
gcc --version                                                                                                                                                                                                        
g++ --version
```

输出类似如下内容：

```text
(oge) ➜  OGE git:(main) ✗ gcc --version
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

(oge) ➜  OGE git:(main) ✗ g++ --version
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### 1.2 安装 vcpkg

克隆 `vcpkg` 并且执行安装

```bash
git clone https://www.github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
```

将以下内容添加到 `~/.bashrc` 或者 `~/.zshrc`

```bash
# >>> vcpkg
export VCPKG_ROOT=<path-to-vcpkg>
export PATH=$VCPKG_ROOT:$PATH
# <<< vcpkg
```

### 1.3 Python 环境配置

创建 `conda` 环境并且安装基础依赖

```bash
conda create -n oge python=3.13
conda activate oge
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

安装 `skrl` 库

```bash
conda activate oge
cd third_party/skrl-1.4.3
pip install -e ".["torch"]" -i https://mirrors.ustc.edu.cn/pypi/simple
```

## 2. 安装 OGE

```bash
conda activate oge
pip install .
```

## 3. 运行 SKRL 训练示例

```bash
python scripts/train.py
```