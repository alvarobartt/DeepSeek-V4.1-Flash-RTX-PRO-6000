FROM vllm/vllm-openai:v0.32.0

RUN python3 -m pip install --no-deps --upgrade flashinfer-python==0.7.1 \
    && python3 -m pip install --no-deps --upgrade flashinfer-cubin==0.7.1 --index-url https://flashinfer.ai/whl \
    && python3 -m pip install --no-deps --upgrade flashinfer-jit-cache==0.7.1 flashinfer-jit-cache-sm120f==0.7.1 --index-url https://flashinfer.ai/whl/cu130

COPY 8-nvidia-rtx-pro-6000/diff.patch /tmp/diff.patch
RUN patch --batch --forward -p1 -d /usr/local/lib/python3.12/dist-packages < /tmp/diff.patch \
    && rm /tmp/diff.patch

RUN mkdir -p /tmp/nccl-tuner/tuner \
    && curl -fL https://raw.githubusercontent.com/NVIDIA/nccl/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/include/plugin/nccl_tuner.h -o /tmp/nccl-tuner/nccl_tuner.h \
    && curl -fL https://raw.githubusercontent.com/NVIDIA/nccl/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/include/nccl_common.h -o /tmp/nccl-tuner/nccl_common.h \
    && for nccl_api in 2 3 4 5 6; do curl -fL https://raw.githubusercontent.com/NVIDIA/nccl/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/include/plugin/tuner/tuner_v${nccl_api}.h -o /tmp/nccl-tuner/tuner/tuner_v${nccl_api}.h; done \
    && g++ -std=c++17 -O3 -shared -fPIC -Wall -Wextra -Werror -I/tmp/nccl-tuner -I/usr/local/cuda/include -I/usr/local/lib/python3.12/dist-packages/nvidia/nccl/include /usr/local/lib/python3.12/dist-packages/vllm/distributed/nccl_tuner.cc -o /usr/local/lib/libvllm-sm120-tuner.so \
    && rm -r /tmp/nccl-tuner

RUN nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC --cudart shared -rdc=true -DPACKET_LANES=16 \
    -gencode arch=compute_120a,code=sm_120a \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/sm120_host_all_reduce.cu \
    -o /usr/local/lib/libvllm-sm120-host-ll.so

ENV NCCL_TUNER_PLUGIN=/usr/local/lib/libvllm-sm120-tuner.so

ENV VLLM_SM120_TARGET_GREEDY=1

ENV VLLM_SM120_DIRECT_DENSE=1
