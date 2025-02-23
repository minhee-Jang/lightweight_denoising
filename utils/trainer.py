import time
import torch
import random
from utils.tester import test_net_by_tensor_patches

def get_log_msg(log_dict):
    msg = ''
    for k, v in log_dict.items():
        msg += '{}: {} '.format(k, v)

    return msg

def print_logs(phase, batch_time, epoch, n_epochs, itr, total_itr, msg):
    print("{} {:.3f}s => Epoch[{}/{}]({}/{}): {}".format(
        phase, batch_time, epoch, n_epochs, itr, total_itr, msg
    ))

def train_net(opt, model, dataloader, is_train=False):
    if is_train:
        print("*** Training phase ***")
        phase = 'training'
        model.train()
    else:
        print("*** Validation phase ***")
        phase = 'validation'
        model.eval()
        
    avg_loss = 0.0
    avg_psnr = 0.0

    start_time = time.time()
    for i, batch in enumerate(dataloader, 1):
        model.set_input(batch)
        if is_train:
            model.optimize_parameters()
        else:
            with torch.no_grad():
                model.set_input(batch)
                model.test()
                model.calc_loss()

        log_msg = get_log_msg(model.get_logs())
        print_logs(phase, time.time() -start_time, opt.epoch, opt.n_epochs, i, len(dataloader), log_msg)

        batch_loss, batch_psnr = model.get_batch_loss_psnr()
        avg_loss += batch_loss
        avg_psnr += batch_psnr

    avg_loss, avg_psnr = avg_loss / i, avg_psnr / i
    # log_epoch_loss("Training", avg_loss, avg_psnr)
    print("===> {} Batch Average Loss: {:.8f}, Batch Average PSNR: {:.8f}".format(phase, avg_loss, avg_psnr))
    return avg_loss, avg_psnr
