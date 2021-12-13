#!/bin/bash -e
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx*_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
