import torch
import torchvision
import argparse
import deepspeed
import ipdb

net = torchvision.models.resnet18()
params = net.parameters()

parser=argparse.ArgumentParser(description='test')

parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()

model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=net, model_parameters=params)

ipdb.set_trace()