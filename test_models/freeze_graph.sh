#!/bin/sh

set -e

TF_VERSION=r1.13

MODEL='models/model.pb'
CHECKPOINT_PATH='models/model'
FREEZED_MODEL='export/freezed_model.pb'
OPTIMIZED_MODEL='export/optimized_model.pb'
INPUT_NODE='input_'
OUTPUT_NODE='LeNetDropout/softmax_linear/Identity'

# Freeze the computational graph with the variables
python3 -m tensorflow.python.tools.freeze_graph --input_graph=$MODEL \
  	--input_checkpoint=$CHECKPOINT_PATH \
  	--output_graph=$FREEZED_MODEL \
  	--output_node_names=$OUTPUT_NODE

# Optimize the model for inference
python3 -m tensorflow.python.tools.optimize_for_inference --input=$FREEZED_MODEL \
	--output=$OPTIMIZED_MODEL \
	--input_names=$INPUT_NODE  \
	--output_names=$OUTPUT_NODE

