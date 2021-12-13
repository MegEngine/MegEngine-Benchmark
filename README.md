# MegEngine-Benchmark

本仓库对 [Models](https://github.com/MegEngine/Models) 与 [basecls](https://github.com/megvii-research/basecls) 中的模型进行基本测速，
输出信息包括：每个 iter 运行时间(单位: ms)，最大显存占用(单位: MiB)，平均 cpu 占用率(top loadavg 的显示值)。

## 使用指南

#### 一键运行自定义的所有 benchmarks

```bash
# 运行单卡 benchmark
./run_benchmark.sh

# 运行单卡+八卡 benchmark
./run_benchmark.sh -d

# 运行单卡+八卡+pytorch benchmark
./run_benchmark.sh -d -t
```

#### 运行指定的 benchmark

运行指定 benchmark 只能获得每个 iter 的运行时间这一项数据，无法自动从外部监控显存/cpu等信息。

- 基本参数

     指定模型和卡数: `run.py -b <benchmark_name> -n <ngpus>`

- 可选参数

     1. 指定 batch_size: `run.py -b <benchmark_name> -n <ngpus> --batch-size <bs>`
     2. 指定训练步数: `run.py -b <benchmark_name> -n <ngpus> --steps <n_steps>`
     3. 使用混合精度训练: `run.py -b <benchmark_name> -n <ngpus> -m mp`
     4. 使用 trace: `run.py -b <benchmark_name> -n <ngpus> -t`
     5. trace 指定使用 symbolic=True 模式: `run.py -b <benchmark_name> -n <ngpus> -t --symbolic`
     6. 使用 dataloader: `run.py -b <benchmark_name> -n <ngpus> --loader`
     7. 使用 preloader(需要先打开 dataloader): `run.py -b <benchmark_name> -n <ngpus> --loader --preload`

- 示例运行

```bash
> ./run.py -b faster_rcnn -n 1 --steps 20
command:  source prepare.sh && python3 ./detection/train_random.py -a faster_rcnn -n 1 -s 20
13 14:12:40 load_serialized_obj_from_url: download to or using cached /home/chenyuanzhao/.cache/megengine/serialized/928d77_resnet50_fbaug_76254_4e14b7d1.pkl
Step 0, Loss (5.559 0.688 0.478 4.394 0.000 ), Time (tot:0.529, data:0.000)
Step 1, Loss (5.552 0.688 0.478 4.386 0.000 ), Time (tot:0.526, data:0.000)
Step 2, Loss (5.542 0.687 0.477 4.377 0.000 ), Time (tot:0.519, data:0.000)
Step 3, Loss (5.532 0.687 0.477 4.368 0.000 ), Time (tot:0.524, data:0.000)
Step 4, Loss (5.522 0.687 0.477 4.358 0.000 ), Time (tot:0.525, data:0.000)
Step 5, Loss (5.513 0.687 0.477 4.349 0.000 ), Time (tot:0.527, data:0.000)
Step 6, Loss (5.503 0.688 0.477 4.338 0.000 ), Time (tot:0.527, data:0.000)
Step 7, Loss (5.492 0.687 0.476 4.328 0.000 ), Time (tot:0.530, data:0.000)
Step 8, Loss (5.484 0.688 0.476 4.319 0.000 ), Time (tot:0.527, data:0.000)
Step 9, Loss (5.470 0.685 0.476 4.308 0.000 ), Time (tot:0.527, data:0.000)
Step 10, Loss (5.462 0.688 0.476 4.298 0.000 ), Time (tot:0.526, data:0.000)
Step 11, Loss (5.448 0.685 0.476 4.288 0.000 ), Time (tot:0.526, data:0.000)
Step 12, Loss (5.438 0.686 0.475 4.276 0.000 ), Time (tot:0.524, data:0.000)
Step 13, Loss (5.428 0.685 0.475 4.267 0.000 ), Time (tot:0.523, data:0.000)
Step 14, Loss (5.417 0.685 0.475 4.257 0.000 ), Time (tot:0.522, data:0.000)
Step 15, Loss (5.403 0.682 0.475 4.246 0.000 ), Time (tot:0.521, data:0.000)
Step 16, Loss (5.393 0.684 0.475 4.234 0.000 ), Time (tot:0.521, data:0.000)
Step 17, Loss (5.385 0.685 0.475 4.225 0.000 ), Time (tot:0.521, data:0.000)
Step 18, Loss (5.374 0.684 0.474 4.213 0.003 ), Time (tot:0.521, data:0.000)
Step 19, Loss (5.361 0.683 0.474 4.200 0.003 ), Time (tot:0.521, data:0.000)
==================== summary ====================
 benchmark: detection
      mode: imperative
    loader:
      arch: faster_rcnn
train_mode: normal
 batchsize: 2
      #GPU: 1
  avg time: 0.521 seconds
```

会发现 run.py 只是对模型做了分发，仍然是进入到特定的模型子目录中去运行对应的 train_random.py。

## 可选的 benchmarks

* shufflenet
* resnet
* faster_rcnn
* atss
* retinanet
* vision_transformer
* [basecls related models](https://github.com/megvii-research/basecls/blob/main/basecls/models/__init__.py)
* torch_shufflenet
* torch_resnet
* torch_vision_transformer
* [timm related models](https://github.com/rwightman/pytorch-image-models)
