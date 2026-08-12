# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND="noninteractive"

# Render headless by default. Override at runtime for GUI use, e.g.:
#   docker run -e QT_QPA_PLATFORM=xcb -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...
ENV QT_QPA_PLATFORM=offscreen

# System libs: OpenGL runtime (pyqtgraph/PyOpenGL), Qt6 runtime (PyQt6 wheels),
# and the Qt6 xcb platform plugin libs for running with a forwarded X display.
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
                    python3-dev \
                    python3-pip \
                    git \
                    build-essential \
                    libgl1 \
                    libegl1 \
                    libopengl0 \
                    libglu1-mesa \
                    libglib2.0-0 \
                    libfontconfig1 \
                    libdbus-1-3 \
                    libsm6 \
                    libxrender1 \
                    libxext6 \
                    libxkbcommon-x11-0 \
                    libxcb-cursor0 \
                    libxcb-icccm4 \
                    libxcb-image0 \
                    libxcb-keysyms1 \
                    libxcb-randr0 \
                    libxcb-render-util0 \
                    libxcb-shape0 \
                    libxcb-xkb1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

COPY . /f1tenth_gym

# arm64: PyQt6 6.7.1 is the newest release with an aarch64 wheel compatible
# with Ubuntu 22.04's glibc 2.35 — newer ones need glibc >= 2.39, so pip
# falls back to the sdist and fails looking for qmake (seen on Apple
# Silicon). x86_64 is unaffected by the constraint.
RUN echo 'pyqt6 == 6.7.1; platform_machine == "aarch64"' > /tmp/pip-constraints.txt && \
    cd /f1tenth_gym && \
    pip3 install -c /tmp/pip-constraints.txt -e .

# Smoke test: step the env and render a frame offscreen. Also bakes the
# default track (downloaded on first use) into the image.
RUN python3 -c "\
import gymnasium as gym; \
import f1tenth_gym; \
env = gym.make('f1tenth_gym:f1tenth-v0', render_mode='rgb_array'); \
env.reset(); \
[env.step(env.action_space.sample()) for _ in range(50)]; \
frame = env.render(); \
assert frame is not None and frame.size > 0; \
print('smoke test OK')"

WORKDIR /f1tenth_gym

ENTRYPOINT ["/bin/bash"]
