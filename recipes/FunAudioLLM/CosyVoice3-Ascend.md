# CosyVoice3 streaming on Ascend

This opt-in profile places the talker and Code2Wav on one logical NPU. It
requires the CosyVoice3 Stage-1 NPUGraph adapter. The generic CUDA-oriented
`cosyvoice3.yaml` remains unchanged; this profile is not selected automatically.

## Start the service

Use an available physical NPU and a downloaded
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` checkpoint:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
COSYVOICE3_STAGE1_NPU_GRAPH=1 \
COSYVOICE3_BATCH_FLOW=1 \
vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --omni \
    --deploy-config vllm_omni/deploy/cosyvoice3_ascend.yaml
```

Every request still needs `ref_audio` and `ref_text`; see the
[CosyVoice3 serving guide](../../docs/user_guide/examples/online_serving/text_to_speech.md#cosyvoice3).

## Configuration

| Setting | Value | Purpose |
| --- | --- | --- |
| `codec_chunk_frames` | 4 | Send the initial token hop early |
| `codec_stream_scale_factor` | 24 | Increase subsequent hops quickly |
| `codec_max_chunk_frames` | 96 | Cap later hops |
| `codec_left_context_frames` | 25 | Retain existing left context |
| `connector_get_sleep_s` | 0.01 | Keep the existing polling interval |
| Stage-0 graph mode | `FULL_DECODE_ONLY` | Capture decode; prefill stays eager |
| Stage-1 `enforce_eager` | `false` | Enable internal Flow and HiFT graphs |
| Stage-1 outer graph mode | `NONE` | Keep request state outside capture |

The nominal hop progression is 4, 96, 96, ...; prompt alignment can change the
first hop. At 25 speech tokens per second, these nominal increments represent
about 160 ms and 3.84 s of speech tokens. They are not direct TTFP predictions.
Larger later chunks reduce decoder invocations but may increase gaps between
audio arrivals. Check streaming continuity for the intended application.

## Warmup and measurement

Graph capture is lazy and exact-shape based, with 8 graphs per region by
default. Warm up with the intended concurrency, not only sequential requests.
Report the first concurrent round separately from steady-state results.
Unseen shapes can still incur capture or eager execution after warmup.

For a public benchmark path, download seed-tts-eval following the
[dataset instructions](../../benchmarks/build_dataset/download_process_data_seedtts.md)
and use the shared benchmark command, for example:

```bash
vllm bench serve --omni \
    --host 127.0.0.1 --port 8000 \
    --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --backend openai-audio-speech --endpoint /v1/audio/speech \
    --dataset-name seed-tts --dataset-path /path/to/seed-tts-eval \
    --seed-tts-locale zh --num-prompts 8 --num-warmups 2 \
    --max-concurrency 8 --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
    --save-result --result-dir ./cosyvoice3-ascend-results
```

This is a suggested public reproduction command, not a claim that it was run
for the historical numbers below. The earlier integration benchmark used a
custom client with a 4.901-second reference and 30 combined reference/target
text tokens, 8 concurrent requests and 2 sequential warmups per invocation.

On one Ascend950PR, historical integration measurements at Omni `6b35bca` plus
the Stage-0 and Stage-1 graph changes compared the prior 6-frame profile with
this 4-to-96 profile:

| Integration configuration | Stable requests | Mean TTFP | Mean RTF |
| --- | --- | --- | --- |
| Prior graph-enabled profile, first hop 6 | 24 | 467.21 ms | 0.5491 |
| Final graph-enabled profile, first hop 4, later cap 96 | 96 | 379.23 ms | 0.2295 |

These are combined configuration results, not an isolated speedup for this
YAML or a benchmark of the latest upstream revision. Generated durations vary.
The final 96 stable requests had 2 TTFP outliers up to 501.26 ms and no RTF
above 0.6; the two first-concurrency rounds averaged 589.49 ms TTFP, with one
RTF outlier above 0.6. No per-request latency guarantee is implied.
WER, speaker similarity, audio continuity, peak memory, other concurrency
levels, and other Ascend products were not validated by these measurements.

Disable internal Stage-1 graphs with `enforce_eager: true` or
`COSYVOICE3_STAGE1_NPU_GRAPH=0`. Capture failure requires restarting the stage;
`ASCEND_LAUNCH_BLOCKING=1` is incompatible with this graph mode.
