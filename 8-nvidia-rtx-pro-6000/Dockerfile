FROM vllm/vllm-openai:v0.32.0

RUN python3 -m pip install --no-deps --upgrade flashinfer-python==0.7.1 \
    && python3 -m pip install --no-deps --upgrade flashinfer-cubin==0.7.1 --index-url https://flashinfer.ai/whl \
    && python3 -m pip install --no-deps --upgrade flashinfer-jit-cache==0.7.1 flashinfer-jit-cache-sm120f==0.7.1 --index-url https://flashinfer.ai/whl/cu130

COPY 8-nvidia-rtx-pro-6000/diff.patch /tmp/diff.patch
RUN patch --batch --forward -p1 -d /usr/local/lib/python3.12/dist-packages < /tmp/diff.patch \
    && rm /tmp/diff.patch
