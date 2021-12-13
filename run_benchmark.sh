#!/bin/bash -e
# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

RUN_DISTRIBUTED=0
RUN_TORCH=0
while getopts ":hdt" opt; do
    case ${opt} in
    h ) # show usage
        echo "run_benchmark usage:"
        echo "    ./run_benchmark -h        Display this help message."
        echo "    ./run_benchmark           Run benchmark."
        echo "    ./run_benchmark -d        Run distributed benchmark."
        echo "    ./run_benchmark -t        Run pytorch benchmark."
        exit 0
        ;;
    d ) # run distributed benchmarks
        RUN_DISTRIBUTED=1
        ;;
    t ) # run pytorch benchmarks
        RUN_TORCH=1
        ;;
    \? ) 
        echo "Usage: cmd [-h] [-d] [-t]"
        ;;
    esac
done
shift $((OPTIND -1))

CUR_DIR="$( cd "$(dirname $0)" >/dev/null 2>&1 ; pwd -P )"
cd ${CUR_DIR}

python3 -m pip install -r requires.txt

BASECLS_INSTALLED=`python3 -m pip list | grep basecls | wc -l`
if [ $BASECLS_INSTALLED -eq 0 ]
then
    echo "install latest basecls"
    git clone https://github.com/megvii-research/basecls.git
    pushd basecls > /dev/null
        python3 -m pip install -r requirements.txt
        cat /dev/null > README.md
        python3 setup.py develop --user
    popd > /dev/null
fi

echo "starting perf..."

FINAL_LOG_FILE=${CUR_DIR}/model.perf.txt
echo "" > ${FINAL_LOG_FILE}
echo "model_name,nr_gpus,use_trace,batch_size,use_loader,use_preloader,train_mode,time_per_iter(ms),max_gpu_usage(MiB),avg_gpu_usage(MiB),avg_cpu_usage" > ${FINAL_LOG_FILE}

# usage: perf_model model_name [n_gpus] [use_trace] [train_mode] [use_symbolic]
function perf_model () {
    BTACHSIZE=0                 # default: 0 (use default batch_size in train_random.py)
    NR_GPUS=1                   # default: use 1 gpu
    USE_TRACE=false             # default: do not use trace
    TRAIN_MODE="normal"         # default: normal train
    USE_SYMBOLIC=false          # default: do not use symbolic
    USE_LOADER=false            # default: do not use loader
    USE_PRELOAD=false           # default: do not use preload
    STEP=${STEP:=200}           # default: number of step is 200

    local OPTIND opt
    while getopts "m:n:b:a:tslp" opt; do
        case ${opt} in
            m )
                echo "model name: ${OPTARG}"
                MODEL_NAME=${OPTARG}
                ;;
            n )
                echo "n_gpu: ${OPTARG}"
                NR_GPUS=${OPTARG}
                ;;
            b )
                echo "batch_size: ${OPTARG}"
                BTACHSIZE=${OPTARG}
                ;;
            t )
                echo "use trace"
                USE_TRACE=true
                ;;
            a )
                echo "train_mode: ${OPTARG}"
                TRAIN_MODE=${OPTARG}
                ;;
            s )
                echo "use symbolic"
                USE_SYMBOLIC=true
                ;;
            l )
                echo "use loader"
                USE_LOADER=true
                ;;
            p )
                echo "use preload"
                USE_PRELOAD=true
                ;;
            * )
                echo "invalid arg: ${opt}"
                exit 1
                ;;
        esac
    done

    TIME_LOG_FILE=/tmp/${MODEL_NAME}.time
    GPU_LOG_FILE=/tmp/${MODEL_NAME}.gpu
    CPU_LOG_FILE=/tmp/${MODEL_NAME}.cpu

    if [ $BTACHSIZE -eq 0 ]; then
        BTACHSIZE=""
    else
        BTACHSIZE="--batch-size ${BTACHSIZE}"
    fi

    if [ $USE_LOADER = true ]; then
        LOADER_COMMAND="--loader"
        if [ $USE_PRELOAD = true ]; then
            PRELOAD_COMMAND="--preload"
        else
            PRELOAD_COMMAND=""
        fi
    else
        LOADER_COMMAND=""
        PRELOAD_COMMAND=""
    fi

    # give cpu some time to rest
    sleep 10

    # empty intermidiate log files
    echo "" > ${TIME_LOG_FILE}
    echo "" > ${GPU_LOG_FILE}
    echo "" > ${CPU_LOG_FILE}

    # monitoring gpu usage in another process
    if [[ -z "$CUDA_VISIBLE_DEVICES" ]]
    then
        SPECIFIC_CARD=""
    else
        SPECIFIC_CARD="-i $CUDA_VISIBLE_DEVICES"
    fi
    while /bin/true; do
        nvidia-smi $SPECIFIC_CARD --query-gpu=memory.used --format="csv,noheader" >> ${GPU_LOG_FILE}
        sleep 0.5
    done &
    gpu_per_pid=$!

    # monitoring cpu usage in another process
    while /bin/true; do
        top -bn 1 | head -n 8 | tail -n 1 >> ${CPU_LOG_FILE}
        sleep 0.5
    done &
    cpu_per_pid=$!

    TRACE=""
    if [ $USE_TRACE = true ]; then
        TRACE="--trace"
    fi
    SYMBOLIC=""
    if [ $USE_SYMBOLIC = true ]; then
        SYMBOLIC="--symbolic"
    fi
    echo "cmd: ./run.py -b ${MODEL_NAME} -n ${NR_GPUS} -m ${TRAIN_MODE} ${TRACE} ${SYMBOLIC} ${BTACHSIZE} ${LOADER_COMMAND} ${PRELOAD_COMMAND} --steps ${STEP}"
    ./run.py -b ${MODEL_NAME} -n ${NR_GPUS} -m ${TRAIN_MODE} ${TRACE} ${SYMBOLIC} ${BTACHSIZE} ${LOADER_COMMAND} ${PRELOAD_COMMAND} --steps ${STEP} | tee ${TIME_LOG_FILE}

    # kill background nvidia-smi/top process after executing
    kill -9 ${gpu_per_pid} >/dev/null 2>&1
    kill -9 ${cpu_per_pid} >/dev/null 2>&1
    skill -9 interpreter

    # process time, gpu memory and cpu usage
    AVG_TIME_RUN=$(awk '/^  avg time:/{print $3}' ${TIME_LOG_FILE})
    AVG_TIME_RUN=$(awk "BEGIN {print $AVG_TIME_RUN * 1000}")
    MAX_GPU_OCCUPIED=$(awk -v max=0 '{if($1>max){res=$1; max=$1}} END {print res}' ${GPU_LOG_FILE})
    AVG_GPU_OCCUPIED=$(awk -v sum=0 '{sum+=$1} END {print sum/NR}' ${GPU_LOG_FILE})
    AVG_CPU_USAGE=$(awk -v sum=0 '{sum+=$9} END {print sum/NR}' ${CPU_LOG_FILE})
    BASE_BATCHES=$(awk '/ batchsize:/{print $2}' ${TIME_LOG_FILE})

    # write data back to final log file
    echo "${MODEL_NAME},${NR_GPUS},${USE_TRACE},${BASE_BATCHES},${USE_LOADER},${USE_PRELOAD},${TRAIN_MODE},${AVG_TIME_RUN},${MAX_GPU_OCCUPIED},${AVG_GPU_OCCUPIED},${AVG_CPU_USAGE}" >> ${FINAL_LOG_FILE}

    # delete intermidiate log file
    rm -f ${TIME_LOG_FILE}
    rm -f ${GPU_LOG_FILE}
    rm -f ${CPU_LOG_FILE}
}

# =============== benchmarks ===============
# usage: perf_model -m model_name [-n n_gpus] [-b batch_size] [-t] [-a] [-s]

# ***** classification *****
perf_model -m shufflenet -n 1
perf_model -m shufflenet -n 1 -l
perf_model -m shufflenet -n 1 -l -p

perf_model -m resnet -n 1

# ***** detection *****
perf_model -m faster_rcnn -n 1
perf_model -m atss -n 1
perf_model -m retinanet -n 1

# ***** transformer *****
perf_model -m vision_transformer -n 1

# ***** pytorch *****
if [ $RUN_TORCH -eq 1 ]
then
    perf_model -m torch_resnet -n 1
    perf_model -m torch_resnet -n 8
    perf_model -m torch_resnet -n 8 -a mp
    perf_model -m torch_resnet -n 8 -b 32 -a qat

    perf_model -m torch_shufflenet -n 1
    perf_model -m torch_shufflenet -n 8
    perf_model -m torch_shufflenet -n 8 -a mp
    perf_model -m torch_shufflenet -n 8 -b 32 -a qat

    perf_model -m torch_vision_transformer -n 1
    perf_model -m torch_vision_transformer -n 8
    perf_model -m torch_vision_transformer -n 8 -a mp
fi

# ***** basecls(megengine) && timm(pytorch) *****
if [ $RUN_DISTRIBUTED -eq 1 ]
then
    perf_model -m shufflenet -n 8
    perf_model -m shufflenet -n 8 -t
    perf_model -m shufflenet -n 8 -l
    perf_model -m shufflenet -n 8 -l -p
    perf_model -m shufflenet -n 8 -a mp
    perf_model -m shufflenet -n 8 -t -a mp
    perf_model -m shufflenet -n 8 -b 32 -a qat
    perf_model -m shufflenet -n 8 -b 32 -t -a qat

    perf_model -m resnet -n 8
    perf_model -m resnet -n 8 -l
    perf_model -m resnet -n 8 -l -p
    perf_model -m resnet -n 8 -t
    perf_model -m resnet -n 8 -a mp
    perf_model -m resnet -n 8 -t -a mp
    perf_model -m resnet -n 8 -b 32 -a qat
    perf_model -m resnet -n 8 -b 32 -t -a qat

    perf_model -m faster_rcnn -n 8
    # FIXME: fix trace for faster_rcnn
    # perf_model -m faster_rcnn -n 8 -t
    perf_model -m faster_rcnn -n 8 -a mp

    perf_model -m atss -n 8
    perf_model -m atss -n 8 -t
    perf_model -m atss -n 8 -a mp
    perf_model -m atss -n 8 -t -a mp

    perf_model -m retinanet -n 8
    perf_model -m retinanet -n 8 -l
    perf_model -m retinanet -n 8 -l -p
    perf_model -m retinanet -n 8 -t
    perf_model -m retinanet -n 8 -a mp
    perf_model -m retinanet -n 8 -t -a mp

    perf_model -m vision_transformer -n 8
    # FIXME: fix trace for vision_transformer
    # perf_model -m vision_transformer -n 8 -t
    perf_model -m vision_transformer -n 8 -a mp

    for MODEL_NAMES in \
        "effnet_b0 efficientnet_b0 32" \
        "effnet_b5 efficientnet_b5 16" \
        "hrnet_w18 hrnet_w18 32" \
        "hrnet_w40 hrnet_w40 32" \
        "mbnetv2_x100 mobilenetv2_100 32" \
        "mbnetv3_small_x100 mobilenetv3_small_100 32" \
        "mbnetv3_large_x100 mobilenetv3_large_100 32" \
        "regnetx_002 regnetx_002 32" \
        "regnetx_160 regnetx_160 32" \
        "regnety_002 regnety_002 32" \
        "regnety_160 regnety_160 32" \
        "vgg11_bn vgg11_bn 32" \
        "vgg16_bn vgg16_bn 16"
        # "basecls_model_name timm_model_name batch_size"
    do
        set -- ${MODEL_NAMES}
        perf_model -m basecls_$1 -n 8 -b $3
        if [ $RUN_TORCH -eq 1 ]
        then
            perf_model -m timm_$2 -n 8 -b $3
        fi
    done
fi

echo "Show benchmark results"
cat ${FINAL_LOG_FILE}
export PYTHONIOENCODING=utf-8
python3 ./show_benchmark_results.py --path ${FINAL_LOG_FILE}
