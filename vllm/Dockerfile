ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0-cu129
FROM ${VLLM_IMAGE}

# The NVIDIA container toolkit mounts these CUDA forward-compat libraries over the
# host driver when the host driver is older than the image's CUDA. They only work
# on datacenter GPUs; on consumer GPUs (e.g. RTX 3090) CUDA then fails with
# "Error 804: forward compatibility was attempted on non supported HW".
# Removing them makes the container use the host driver directly.
RUN rm -rf /usr/local/cuda*/compat
RUN pip install --no-cache-dir num2words # for smolvlm2-2.2B-instruct