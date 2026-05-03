# Cluster Resource Snapshot

## Known cluster inventory

| Node | GPU inventory | Shared storage | Notes |
| --- | --- | --- | --- |
| Sui-3-Wu | 8 x RTX 3090 24GB | `/NAS` | Mostly idle in this snapshot except GPU 4. |
| Sui-5-Wu | 8 x RTX 3090 24GB | `/NAS` | SSH/nvidia-smi query timed out during snapshot. |
| Tang-1-Wu | 8 x A40 46GB | `/NAS` | Mostly busy; GPU 2 partially free. |
| Tang-2-Wu | 8 x A40 46GB | `/NAS` | Busy but usable for CPU/validation. |
| Tang-3-Wu | 8 x A40 46GB | `/NAS` | GPU 2 appears free; GPUs 6-7 have high memory reserved but low utilization. |
| Song-1-Wu | 7 x A100 80GB | `/NAS` | Best node for current learned spot-checks; all GPUs effectively idle. |
| Song-3-Wu | 7/8 reported x A100 80GB | `/NAS` | Mostly busy in this snapshot. |

## Snapshot details

### Sui-3-Wu

```text
0, NVIDIA GeForce RTX 3090, 24576 MiB, 3 MiB, 0 %
1, NVIDIA GeForce RTX 3090, 24576 MiB, 3 MiB, 0 %
2, NVIDIA GeForce RTX 3090, 24576 MiB, 3 MiB, 0 %
3, NVIDIA GeForce RTX 3090, 24576 MiB, 3 MiB, 0 %
4, NVIDIA GeForce RTX 3090, 24576 MiB, 5137 MiB, 100 %
5, NVIDIA GeForce RTX 3090, 24576 MiB, 3 MiB, 0 %
6, NVIDIA GeForce RTX 3090, 24576 MiB, 3 MiB, 0 %
7, NVIDIA GeForce RTX 3090, 24576 MiB, 3 MiB, 0 %
```

### Tang-1-Wu

```text
0, NVIDIA A40, 46068 MiB, 43518 MiB, 100 %
1, NVIDIA A40, 46068 MiB, 42748 MiB, 100 %
2, NVIDIA A40, 46068 MiB, 21828 MiB, 0 %
3, NVIDIA A40, 46068 MiB, 42748 MiB, 100 %
4, NVIDIA A40, 46068 MiB, 43516 MiB, 100 %
5, NVIDIA A40, 46068 MiB, 42748 MiB, 100 %
6, NVIDIA A40, 46068 MiB, 42748 MiB, 100 %
7, NVIDIA A40, 46068 MiB, 43518 MiB, 100 %
```

### Tang-2-Wu

```text
0, NVIDIA A40, 46068 MiB, 30228 MiB, 76 %
1, NVIDIA A40, 46068 MiB, 29532 MiB, 77 %
2, NVIDIA A40, 46068 MiB, 29764 MiB, 81 %
3, NVIDIA A40, 46068 MiB, 29302 MiB, 84 %
4, NVIDIA A40, 46068 MiB, 41782 MiB, 100 %
5, NVIDIA A40, 46068 MiB, 41994 MiB, 100 %
6, NVIDIA A40, 46068 MiB, 42998 MiB, 95 %
7, NVIDIA A40, 46068 MiB, 43468 MiB, 94 %
```

### Tang-3-Wu

```text
0, NVIDIA A40, 46068 MiB, 41782 MiB, 100 %
1, NVIDIA A40, 46068 MiB, 42274 MiB, 100 %
2, NVIDIA A40, 46068 MiB, 7 MiB, 0 %
3, NVIDIA A40, 46068 MiB, 41782 MiB, 100 %
4, NVIDIA A40, 46068 MiB, 41782 MiB, 100 %
5, NVIDIA A40, 46068 MiB, 41782 MiB, 100 %
6, NVIDIA A40, 46068 MiB, 45130 MiB, 0 %
7, NVIDIA A40, 46068 MiB, 45130 MiB, 0 %
```

### Song-1-Wu

```text
0, NVIDIA A100-SXM4-80GB, 81920 MiB, 1221 MiB, 0 %
1, NVIDIA A100-SXM4-80GB, 81920 MiB, 4 MiB, 0 %
2, NVIDIA A100-SXM4-80GB, 81920 MiB, 4 MiB, 0 %
3, NVIDIA A100-SXM4-80GB, 81920 MiB, 4 MiB, 0 %
4, NVIDIA A100-SXM4-80GB, 81920 MiB, 4 MiB, 0 %
5, NVIDIA A100-SXM4-80GB, 81920 MiB, 4 MiB, 0 %
6, NVIDIA A100-SXM4-80GB, 81920 MiB, 4 MiB, 0 %
```

### Song-3-Wu

```text
0, NVIDIA A100-SXM4-80GB, 81920 MiB, 63337 MiB, 87 %
1, NVIDIA A100-SXM4-80GB, 81920 MiB, 63343 MiB, 93 %
2, NVIDIA A100-SXM4-80GB, 81920 MiB, 72823 MiB, 0 %
3, NVIDIA A100-SXM4-80GB, 81920 MiB, 63339 MiB, 98 %
4, NVIDIA A100-SXM4-80GB, 81920 MiB, 50917 MiB, 99 %
5, NVIDIA A100-SXM4-80GB, 81920 MiB, 51199 MiB, 99 %
6, NVIDIA A100-SXM4-80GB, 81920 MiB, 50743 MiB, 99 %
7, NVIDIA A100-SXM4-80GB, 81920 MiB, 51080 MiB, 99 %
```

## Node choice for P6.7

- Oracle sanity and CPU validation: `Tang-2-Wu` is sufficient.
- Long25 k=16 spot-checks: use `Song-1-Wu`, preferably GPUs 1 and 2.
- Avoid `Song-3-Wu` and busy A40 nodes for learned reruns unless the snapshot changes.
