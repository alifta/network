#!/bin/bash

module load python/3.7
source ~/py37_jupyter/bin/activate

if [ $# -eq 0 ]
then
  echo "No arguments is passed"
  salloc --time=1:0:0 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=2048M --account=def-yuanzhu srun ~/py37_jupyter/bin/notebook.sh
else
  echo "Creating a jupyter job for $1 hour"
  salloc --time=$1:0:0 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=2048M --account=def-yuanzhu srun ~/py37_jupyter/bin/notebook.sh
fi