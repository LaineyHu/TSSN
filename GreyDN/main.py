import torch
import random
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os

os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_id

torch.manual_seed(args.seed)  #promise to reproduce the result with same seed
random.seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

