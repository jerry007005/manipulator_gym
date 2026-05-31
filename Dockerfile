FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=America/Los_Angeles
ARG USER_ID=robonet
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]


RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-cffi \
    vim \
    wget \
    curl \
    lsb-release \
    sudo \
    android-tools-adb \
    libglew-dev \
    patchelf \
    libosmesa6-dev \
    v4l-utils \
    keyboard-configuration \
    tzdata \
    unzip \
    ffmpeg \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN adduser --disabled-password --gecos '' ${USER_ID} \
    && adduser ${USER_ID} sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER_ID}
WORKDIR /home/${USER_ID}


USER root
RUN apt-get update && apt-get install -y curl lsb-release gnupg2 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y \
        ros-jazzy-ros-base \
        ros-jazzy-ament-cmake \
        ros-jazzy-ament-lint-common \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-moveit \
        ros-jazzy-moveit-configs-utils \
        python3-rosdep \
        python3-vcstool \
        python3-colcon-common-extensions \
    && rosdep init \
    && rosdep update \
    && apt-get install -y libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${USER_ID}

RUN mkdir -p ~/ros2_ws/src
WORKDIR /home/${USER_ID}/ros2_ws/src
RUN git clone -b jazzy https://github.com/TrossenRobotics/trossen_arm_ros.git


USER root
WORKDIR /home/${USER_ID}/ros2_ws
RUN vcs import src < src/trossen_arm_ros/dependencies.repos \
    && rosdep install --from-paths src --ignore-src -r -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    
WORKDIR /home/${USER_ID}/ros2_ws/src/trossen_arm_ros/trossen_arm_hardware
# 拷贝修好的 trossen_arm_hardware 文件到容器里
COPY interface.hpp include/trossen_arm_hardware/interface.hpp
COPY interface.cpp src/interface.cpp


USER ${USER_ID}
WORKDIR /home/${USER_ID}/ros2_ws
RUN source /opt/ros/jazzy/setup.bash \
    && colcon build --symlink-install

# RUN python3 -m pip install --upgrade pip \
#     && pip install moveit-configs-utils

# # USER ${USER_ID}
# WORKDIR /home/${USER_ID}
# RUN git clone https://github.com/rail-berkeley/manipulator_gym.git

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
