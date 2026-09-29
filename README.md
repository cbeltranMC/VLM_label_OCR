# VLM_label_OCR

Batch inference over a folder of images with a Hugging Face vision-language model
served by [vLLM](https://github.com/vllm-project/vllm), all in Docker Compose.

- **`vllm`**: an OpenAI-compatible vLLM server (a thin layer on top of the official
  `vllm/vllm-openai` image, see [Older NVIDIA drivers](#older-nvidia-drivers)) hosting the model (default
  [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)).
  It keeps running, so later runs skip reloading the model.
- **`client`**: sends every image in `input/` to the server along with the prompt in
  `prompts/prompt.txt`, then writes the results to `output/`.

## Requirements

- An NVIDIA GPU, a driver, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker with Compose v2

## Quick start

```bash
cp .env.example .env               # edit settings if needed
cp /path/to/my/images/* input/     # or point INPUT_DIR in .env at another folder
docker compose up -d vllm          # start the model server (first run downloads the model)
docker compose run --rm client     # open a shell in a fresh client container
python run_inference.py            # inside the container: process all images in /data/input
```

To follow the server while it downloads and loads the model (the script fails with
a connection error until it's ready):

```bash
docker compose logs -f vllm
```

`docker compose run` is interactive (like `docker run -it`) and starts the server
first if it isn't running. `--rm` deletes the container when you `exit`, so anything
installed by hand is lost, but your scripts and results live in mounted host folders.
`client/` is mounted at `/app`, so you can edit or add scripts on the host and run them right
away. Inside the container, images are at `/data/input`, results go to
`/data/output`, and the prompt is `/data/prompt.txt`. The server is at
`$VLLM_URL` (`http://vllm:8000/v1`). The script's options (`--input-dir`,
`--concurrency`, …) are listed by `python run_inference.py --help`. If you add a
Python dependency, put it in `client/requirements.txt` and run
`docker compose build client`.

To stop everything and free the GPU:

```bash
docker compose down
```

## Model download & cache

The server mounts your host Hugging Face cache (`~/.cache/huggingface`; change it with
`HF_CACHE_DIR`). On the first start vLLM downloads the model there. Later starts, and
any other Hugging Face tool on this machine, reuse the same files without downloading
again. The container runs as root, so the files it downloads into the cache are
owned by root.

## Changing the model

Set `MODEL_ID` in `.env` to any vision-language model
[supported by vLLM](https://docs.vllm.ai/en/latest/models/supported_models.html), then restart:

```bash
docker compose up -d vllm
```

The client asks the server which model is loaded, so you don't need to change
anything else. For larger models you may need to adjust `MAX_MODEL_LEN` and
`GPU_MEMORY_UTILIZATION`. Gated models also need `HF_TOKEN`.

## Changing the prompt

Edit `prompts/prompt.txt`. It's read on every client run, so you don't need to
restart or rebuild anything. To keep several prompts, point `PROMPT_FILE` in `.env` at
another file.

## Output

Each run writes three files to `output/`:

- `results_<timestamp>.jsonl`: one JSON object per image, written as each image
  finishes, so partial results survive a crash
- `results_<timestamp>.csv`: the same rows sorted by filename
- `summary_<timestamp>.json`: speed and GPU stats for the whole run (see
  [Comparing models](#comparing-models))

| field | meaning |
|---|---|
| `file` | path relative to the input folder (subfolders are included) |
| `output` | model response |
| `error` | error message if this image failed, otherwise empty |
| `seconds` | request latency |
| `ttft_seconds` | time to first token (image preprocessing + vision encoder + prefill) |
| `decode_tokens_per_s` | generation speed after the first token |
| `prompt_tokens` / `completion_tokens` | token usage |

The client exits with a non-zero status if any image failed.

## Comparing models

Every run saves a `summary_<timestamp>.json` file. To put all runs side by side
(this also writes `output/comparison.csv`):

```bash
python compare_runs.py          # inside the client container
```

| summary field | meaning |
|---|---|
| `images_per_s`, `output_tokens_per_s` | throughput over the whole run |
| `latency_s_*`, `ttft_s_*`, `decode_tps_*` | per-image mean / p50 / p95 |
| `gpu_mem_baseline_mib` / `gpu_mem_peak_mib` | GPU memory used before / at most during the run (whole device) |
| `gpu_util_mean_pct`, `gpu_power_mean_w`, `gpu_energy_j` | GPU load and energy during the run |
| `kv_cache_peak_tokens` / `kv_cache_capacity_tokens` | KV cache the run actually needed / what vLLM allocated |

Use `--tag` to label a run (`python run_inference.py --tag "eager, kv 1G"`).

**GPU memory is only comparable with a fixed KV cache.** By default vLLM reserves
`GPU_MEMORY_UTILIZATION` of the card up front and fills the rest with KV cache,
so every model shows about the same memory. To see what a model really needs, set
the KV cache size explicitly and restart the server:

```bash
VLLM_EXTRA_ARGS="--kv-cache-memory-bytes=1G --enforce-eager" docker compose up -d vllm
```

Then `gpu_mem_baseline_mib` is roughly weights + profiled activations + 1 GiB of KV
cache. vLLM's startup log shows the breakdown (`docker compose logs vllm | grep -E
"Model loading took|Actual usage"`). `--enforce-eager` skips CUDA graphs: it saves
memory (about 1 GiB here) but decodes slower. Keep `MAX_MODEL_LEN`,
`MAX_IMAGE_PIXELS`, `CONCURRENCY` and the input images the same across models.

## Configuration

All settings live in `.env` (see [`.env.example`](.env.example) for the defaults):

| variable | purpose |
|---|---|
| `MODEL_ID` | Hugging Face model id |
| `HF_TOKEN` | token for gated/private models |
| `VLLM_IMAGE` | base vLLM image; rebuild with `docker compose build vllm` after changing it |
| `HF_CACHE_DIR` | host Hugging Face cache folder |
| `VLLM_PORT` | host port for the API (`http://localhost:8000/v1`) |
| `MAX_MODEL_LEN` | max context (image + prompt + output tokens) |
| `GPU_MEMORY_UTILIZATION` | share of VRAM vLLM may use |
| `VLLM_EXTRA_ARGS` | extra `vllm serve` flags, e.g. `--kv-cache-memory-bytes=1G --enforce-eager` |
| `INPUT_DIR` / `OUTPUT_DIR` | host folders for images and results |
| `PROMPT_FILE` | prompt text file |
| `MAX_IMAGE_PIXELS` | images are downscaled to at most this many pixels (~1 token per 32×32 px for Qwen3-VL) |
| `MAX_TOKENS` / `TEMPERATURE` | generation settings |
| `CONCURRENCY` | parallel requests (vLLM batches them on the GPU) |
| `GPU_INDEX` | GPU the client monitors for the run summary |
| `HOST_UID` / `HOST_GID` | owner of the output files |

Supported image types: jpg, jpeg, png, bmp, tif, tiff, webp. Images are rotated
according to their EXIF orientation before they're sent.

Because the server exposes an OpenAI-compatible API on `localhost:8000`, you can
also call it from other tools while it's running.

## Older NVIDIA drivers

This setup was tested on an RTX 3090 with driver 535 (CUDA 12.2), which is older
than the CUDA 12.9 the vLLM image is built for. Three workarounds make that
combination run. On a driver ≥ 575 they're harmless.

1. **`NVIDIA_DISABLE_REQUIRE=1`** skips the image's `cuda>=12.9` driver check.
   CUDA 12.x programs run on any 12.x driver ≥ 525.
2. **[`vllm/Dockerfile`](vllm/Dockerfile) deletes the CUDA forward-compat libraries**
   from the image. Otherwise the NVIDIA container toolkit mounts them over the host
   driver, and on consumer GPUs CUDA fails with `Error 804: forward compatibility was
   attempted on non supported HW`.
3. **`VLLM_USE_FLASHINFER_SAMPLER=0`** switches off FlashInfer's sampling kernels,
   which fail with `device kernel image is invalid` on a 12.2 driver.

About image tags: plain `vllm/vllm-openai:vX` tags (v0.20+) are built for CUDA 13 and
need driver ≥ 580. `v0.30.0-cu129` is broken because it ships a CUDA 13 PyTorch, so
this repo uses `v0.29.0-cu129`. Updating the NVIDIA driver to ≥ 580 lets you use
plain tags and drop these workarounds.
