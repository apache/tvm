#!/usr/bin/env bash

declare -a models=(
"resnet18_v1"
"resnet34_v1"
"resnet50_v1"
"resnet101_v1"
"resnet152_v1"
"resnet18_v2"
"resnet34_v2"
"resnet50_v2"
"resnet101_v2"
"resnet152_v2"
"vgg11"
"vgg13"
"vgg16"
"vgg19"
"vgg11_bn"
"vgg13_bn"
"vgg16_bn"
"vgg19_bn"
"alexnet"
"densenet121"
"densenet161"
"densenet169"
"densenet201"
"squeezenet1.0"
"squeezenet1.1"
"inceptionv3"
"mobilenet1.0"
"mobilenet0.75"
"mobilenet0.5"
"mobilenet0.25"
"mobilenetv2_1.0"
"mobilenetv2_0.75"
"mobilenetv2_0.5"
"mobilenetv2_0.25")

for model_name in "${models[@]}"
do
    python run_tvm.py --network="$model_name" --compile
done

for model_name in "${models[@]}"
do
    python run_tvm.py --network="$model_name" --compile --ext-accel=tensorrt
done

for model_name in "${models[@]}"
do
    python run_tvm.py --network="$model_name" --run --ext-accel=tensorrt
done
