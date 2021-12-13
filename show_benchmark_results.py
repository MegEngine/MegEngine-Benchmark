# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import time
import argparse
import csv

from tabulate import tabulate
from rich.console import Console
from rich.table import Table

parser = argparse.ArgumentParser(description="show benchmark results")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="results file path",
)
parser.add_argument(
    "--use-rich",
    action='store_true',
    default=False,
    help="use rich to display (default: False, use tabulate)",
)

args = parser.parse_args()

def print_table_with_rich(args):
    input_file = csv.DictReader(open(args.path, "r"))

    fields = ['model_name', 'nr_gpus', 'use_trace', 'batch_size', 'use_loader', 'use_preloader', 'train_mode', 'time_per_iter(ms)', 'max_gpu_usage(MiB)', 'avg_cpu_usage']
    table = Table(title="MegEngine Benchmarks ({})".format(time.ctime()))

    for field in fields:
        table.add_column(field, justify='right', style='cyan', no_wrap=True)

    for row in input_file:
        data = []
        model_name = row["model_name"]
        for field in fields:
            if field == "use_trace":
                if row[field] == "true":
                    data.append("trace(symbolic=True, sublinear=True)")
                else:
                    data.append("imperative")
            else:
                data.append(row[field])
        table.add_row(*data)

    console = Console()
    console.print(table)


def print_table_with_tabulate(args):
    input_file = csv.DictReader(open(args.path, "r"))

    header = ['model_name', 'nr_gpus', 'use_trace', 'batch_size', 'use_loader', 'use_preloader', 'train_mode', 'time_per_iter(ms)', 'max_gpu_usage(MiB)', 'avg_cpu_usage']
    table = []

    for row in input_file:
        data = []
        model_name = row["model_name"]
        for field in header:
            if field == "use_trace":
                if row[field] == "true":
                    data.append("trace(symbolic=True, sublinear=True)")
                else:
                    data.append("imperative")
            else:
                data.append(row[field])
        table.append(data)
    print(tabulate(table, header, tablefmt="github"))

if __name__ == "__main__":
    if args.use_rich:
        print_table_with_rich(args)
    else:
        print_table_with_tabulate(args)
