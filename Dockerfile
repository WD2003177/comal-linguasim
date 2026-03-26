# 使用包含 Python 3.7 的旧版 Miniconda
FROM continuumio/miniconda3:4.7.12
MAINTAINER Fangyu Wu (fangyuwu@berkeley.edu)

# =========================================================================
# 【底层环境修复】解决 Debian Buster 源失效 + OpenJDK 目录缺失
# =========================================================================
RUN echo "deb http://archive.debian.org/debian buster main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security buster/updates main contrib non-free" >> /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until && \
    mkdir -p /usr/share/man/man1

# =========================================================================
# 【系统依赖安装】替换 JDK 8 -> 11，安装编译工具
# =========================================================================
RUN apt-get update && \
    apt-get install -y \
    vim git build-essential cmake swig libgdal-dev libxerces-c-dev \
    libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev openjdk-11-jdk \
    apt-utils && \
    pip install -U "pip<21.0"

# =========================================================================
# 【Python 依赖预装】关键步骤
# 1. 强制安装二进制包，避开源码编译
# 2. 提前装好 numpy, scipy, tensorflow, gym 等重型包
# =========================================================================
RUN cd ~ && \
    conda install -y python=3.7 opencv && \
    pip install "numpy==1.16.6" "scipy==1.4.1" "tensorflow==1.15.0" \
                "gym==0.10.5" "pandas" "matplotlib" "lxml" "joblib" "pyglet" \
                "boto3" "networkx" "imutils" "pyproj" "shapely"

# =========================================================================
# 【Flow 安装】暴力修改版
# 1. 克隆代码
# 2. 用 sed 删除 setup.py 中所有关于 numpy 的限制行
# 3. 使用 --no-deps 参数安装，禁止 pip 检查和重新安装依赖
# =========================================================================
RUN cd ~ && \
    git clone https://github.com/flow-project/flow.git && \
    cd flow && \
    git checkout v0.3.0 && \
    # 暴力删除 setup.py 里包含 numpy 的行，防止 pip 重新安装它
    sed -i '/numpy/d' setup.py && \
    # 尝试删除 requirements.txt 里的 numpy (如果存在)
    (sed -i '/numpy/d' requirements.txt || true) && \
    # 强行安装，不再检查依赖
    pip install --no-deps -e .

# =========================================================================
# 【SUMO 编译】(这一步在 M4 上模拟 x86 会非常慢，预计 20 分钟+，请耐心)
# =========================================================================
RUN cd ~ && \
    git clone --recursive https://github.com/eclipse/sumo.git && \
    cd sumo && \
    git checkout cbe5b73 && \
    mkdir build/cmake-build && \
    cd build/cmake-build && \
    cmake ../.. && \
    make -j$(nproc)

# Ray/RLlib
RUN cd ~ && \
    pip install ray==0.6.2 psutil
    
# 环境变量设置
RUN echo 'export SUMO_HOME="$HOME/sumo"' >> ~/.bashrc && \
    echo 'export PATH="$HOME/sumo/bin:$PATH"' >> ~/.bashrc && \
    echo 'export PYTHONPATH="$HOME/sumo/tools:$PYTHONPATH"' >> ~/.bashrc