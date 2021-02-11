#!/bin/bash

DATASETS=( 'wnut2016' 'fin' 'jnlpba' 'bc2gm' 'bionlp09' 'bionlp11epi')
STRATEGIES=( 'rand' 'pos' 'entropy' 'adv' )
DIR=/media/trang/data/1Working/emnlp2020
for TASK in {0..5};do
  for STRATEGY in {0..3}; do
    TASK_NAME=${DATASETS[TASK]}
    STRATEGY_NAME=${STRATEGIES[STRATEGY]}
    cat $DIR/dt/"dt-"$TASK"-"$STRATEGY-*.out | grep "^\[CLS\]" > $DIR/dt/"dt-"$TASK_NAME"-"$STRATEGY_NAME'.log'
    cat $DIR/dt-500k/"dt-500k-"$TASK"-"$STRATEGY-*.out | grep "^\[CLS\]" > $DIR/dt-500k/"dt-500k-"$TASK_NAME"-"$STRATEGY_NAME'.log'
#    cat $DIR/sda/"sda-500k-"$TASK"-"$STRATEGY-*.out | grep "^\[CLS\]" > $DIR/sda/"sda-500k-"$TASK_NAME"-"$STRATEGY_NAME'.log'
  done
done