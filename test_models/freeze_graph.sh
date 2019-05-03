#!/bin/bash

set -e

TF_VERSION=r1.13

MODEL='models/model.pb'
CHECKPOINT_PATH='models/model'
FREEZED_MODEL='export/freezed_model.pb'
OPTIMIZED_MODEL='export/optimized_model.pb'
INPUT_NODE='input_'
OUTPUT_NODE='LeNetDropout/softmax_linear/Identity'

# Freeze the computational graph with the variables
wget -q "https://raw.githubusercontent.com/tensorflow/tensorflow/${TF_VERSION}/tensorflow/python/tools/freeze_graph.py"
python3 freeze_graph.py --input_graph=$MODEL \
  	--input_checkpoint=$CHECKPOINT_PATH \
  	--output_graph=$FREEZED_MODEL \
  	--output_node_names=$OUTPUT_NODE

# Optimize the model for inference
wget -q "https://raw.githubusercontent.com/tensorflow/tensorflow/${TF_VERSION}/tensorflow/python/tools/optimize_for_inference.py"
python3 optimize_for_inference.py --input=$FREEZED_MODEL \
	--output=$OPTIMIZED_MODEL \
	--input_names=$INPUT_NODE  \
	--output_names=$OUTPUT_NODE

# Remove temporary files
rm freeze_graph.py* optimize_for_inference.py* || true


