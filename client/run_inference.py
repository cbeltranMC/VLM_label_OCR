"""Run a VLM served by vLLM over every image in a folder.

Writes one row per image to results_<timestamp>.jsonl and .csv in the output folder.
"""

import argparse
import asyncio
import base64
import csv
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CSV_FIELDS = ["file", "output", "error", "seconds", "prompt_tokens", "completion_tokens"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="/data/input")
    p.add_argument("--output-dir", default="/data/output")
    p.add_argument("--prompt-file", default="/data/prompt.txt")
    p.add_argument("--url", default=os.getenv("VLLM_URL", "http://localhost:8000/v1"))
    p.add_argument("--max-image-pixels", type=int, default=int(os.getenv("MAX_IMAGE_PIXELS", 3_000_000)))
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", 1024)))
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", 0.0)))
    p.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", 8)))
    return p.parse_args()


def find_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def encode_image(path: Path, max_pixels: int) -> str:
    """Return a JPEG data URL, rotated per EXIF and downscaled to at most max_pixels."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if img.width * img.height > max_pixels:
            scale = (max_pixels / (img.width * img.height)) ** 0.5
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def process(client, model, prompt, path, rel_name, args, sem):
    async with sem:
        start = time.perf_counter()
        row = {"file": rel_name, "output": None, "error": None,
               "prompt_tokens": None, "completion_tokens": None}
        try:
            data_url = await asyncio.to_thread(encode_image, path, args.max_image_pixels)
            resp = await client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            row["output"] = resp.choices[0].message.content
            row["prompt_tokens"] = resp.usage.prompt_tokens
            row["completion_tokens"] = resp.usage.completion_tokens
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        row["seconds"] = round(time.perf_counter() - start, 2)
        return row


async def main():
    args = parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    prompt = Path(args.prompt_file).read_text().strip()

    images = find_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    client = AsyncOpenAI(base_url=args.url, api_key="EMPTY", timeout=600)
    model = (await client.models.list()).data[0].id
    print(f"Model: {model}\nImages: {len(images)}\nPrompt: {prompt}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"results_{stamp}.jsonl"
    csv_path = output_dir / f"results_{stamp}.csv"

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [process(client, model, prompt, p, str(p.relative_to(input_dir)), args, sem) for p in images]

    start = time.perf_counter()
    rows, failed = [], 0
    with jsonl_path.open("w") as jf:
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            row = await coro
            rows.append(row)
            jf.write(json.dumps({"model": model, **row}, ensure_ascii=False) + "\n")
            jf.flush()
            status = "ERROR " + row["error"] if row["error"] else f"{row['seconds']}s"
            failed += bool(row["error"])
            print(f"[{i}/{len(images)}] {row['file']}: {status}")

    rows.sort(key=lambda r: r["file"])
    with csv_path.open("w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - start
    print(f"\nDone: {len(rows) - failed} ok, {failed} failed in {elapsed:.1f}s")
    print(f"Results: {jsonl_path}\n         {csv_path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
