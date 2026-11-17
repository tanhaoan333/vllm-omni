import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', required=True)
parser.add_argument('--warmups', type=int, default=0)
parser.add_argument('--requests', type=int, default=2)
args = parser.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
body = {
    'model': 'hunyuanimage3-mxfp8',
    'prompt': '一座未来城市，清晨的阳光照亮玻璃摩天大楼，蓝色天空中有几艘飞行汽车，街道上有人们行走，河流穿过城市，写实摄影风格，画面清晰，构图自然，不要文字。',
    'size': '1024x1024', 'bot_task': 'think',
    'use_system_prompt': 'en_unified', 'num_inference_steps': 8,
    'guidance_scale': 2.5, 'seed': 42, 'return_stage_metrics': True,
}
payload = json.dumps(body, ensure_ascii=False, sort_keys=True)
(out / 'request.json').write_text(payload, encoding='utf-8')
rows = []
with httpx.Client(trust_env=False, timeout=1800) as client:
    for i in range(-args.warmups, args.requests):
        start = time.perf_counter()
        first = None
        chunks = []
        with client.stream('POST', 'http://127.0.0.1:8011/v1/images/generations', json=body) as response:
            for chunk in response.iter_bytes():
                if first is None:
                    first = time.perf_counter() - start
                chunks.append(chunk)
        end = time.perf_counter() - start
        response_body = b''.join(chunks)
        prefix = f'request_{i:04d}' if i >= 0 else f'warmup_{-i}'
        (out / f'{prefix}_response.json').write_bytes(response_body)
        if response.status_code != 200:
            raise RuntimeError(response_body.decode()[:2000])
        result = json.loads(response_body)
        hashes = []
        for n, data in enumerate(result['data']):
            image = base64.b64decode(data['b64_json'])
            (out / f'{prefix}_{n}.png').write_bytes(image)
            hashes.append(hashlib.sha256(image).hexdigest())
        metrics = result.get('metrics', {})
        stages = metrics.get('stage_metrics', {})
        row = {
            'index': i, 'e2e_s': end, 'http_first_body_s': first,
            'ar_s': stages.get('0', {}).get('stage_gen_time_ms', 0) / 1000,
            'dit_s': stages.get('1', {}).get('stage_gen_time_ms', 0) / 1000,
            'ar_ttft_ms': stages.get('0', {}).get('vllm_ttft_ms'),
            'ar_tpot_ms': stages.get('0', {}).get('vllm_tpot_ms'),
            'ar_tokens_in': stages.get('0', {}).get('num_tokens_in'),
            'ar_tokens_out': stages.get('0', {}).get('num_tokens_out'),
            'image_sha256': hashes,
        }
        print(json.dumps(row), flush=True)
        if i >= 0:
            rows.append(row)
            (out / 'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
summary = {'requests': len(rows), 'warmups': args.warmups,
           'request_sha256': hashlib.sha256(payload.encode()).hexdigest()}
for key in ['e2e_s', 'http_first_body_s', 'ar_s', 'dit_s', 'ar_ttft_ms', 'ar_tpot_ms']:
    values = [r[key] for r in rows if r[key] is not None]
    summary[key] = sum(values)/len(values) if values else None
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
