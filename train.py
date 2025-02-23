import torch

from utils.trainer import train_net
from data import create_dataset
from models import create_model

from options.train_options import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataloader['train'])    # get the number of images in the dataset.
    
    print('\nThe number of training iteration = {}\n'.format(dataset_size))

    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    best_psnr = 0
    start_epoch = opt.epoch 
    if start_epoch == 1:
        with open(opt.log_file, mode='w') as f:
            f.write("epoch,train_loss,train_psnr,valid_loss,valid_psnr,1sigmax1,1sigmay1,1sigmaz1,1color1,1sigmax2,1sigmay2,1sigmaz2,1color2,1sigmax3,1sigmay3,1sigmaz3,1color3,2sigmax1,2sigmay1,2sigmaz1,2color1,2sigmax2,2sigmay2,2sigmaz2,2color2,2sigmax3,2sigmay3,2sigmaz3,2color3,3sigmax1,3sigmay1,3sigmaz1,3color1,3sigmax2,3sigmay2,3sigmaz2,3color2,3sigmax3,3sigmay3,3sigmaz3,3color3\n")
            #f.write("epoch,train_loss,train_psnr,valid_loss,valid_psnr,1sigmax1,1sigmay1,1sigmaz1,1color1,1sigmax2,1sigmay2,1sigmaz2,1color2,1sigmax3,1sigmay3,1sigmaz3,1color3,2sigmax1,2sigmay1,2sigmaz1,2color1,2sigmax2,2sigmay2,2sigmaz2,2color2,2sigmax3,2sigmay3,2sigmaz3,2color3,3sigmax1,3sigmay1,3sigmaz1,3color1,3sigmax2,3sigmay2,3sigmaz2,3color2,3sigmax3,3sigmay3,3sigmaz3,3color3\n")
 

    for epoch in range(start_epoch, opt.n_epochs+1):
        train_loss, train_psnr = train_net(opt, model, dataloader['train'], is_train=True)
        valid_loss, valid_psnr = train_net(opt, model, dataloader['validation'], is_train=False)

        print('saving the latest model (epoch {}, total_iters {})'.format(epoch, opt.n_epochs))
        model.remove_networks('lates')
        model.save_networks('latest', epoch, valid_loss, valid_psnr)
        # model.save_networks(epoch, epoch, valid_loss)
        if opt.save_epoch_freq != -1 and epoch % opt.save_epoch_freq == 0:
            print('saving the model (epoch {}, total_iters {}) with frequency {}'.format(epoch, opt.n_epochs, opt.save_epoch_freq))
            model.save_networks(epoch, epoch, valid_loss, valid_psnr)
        if valid_psnr > best_psnr:
            print('saving the best model (epoch {}, total_iters {})'.format(epoch, opt.n_epochs))
            model.remove_networks('best')
            model.save_networks('best', epoch, valid_loss, valid_psnr)
            best_psnr = valid_psnr
        # if opt.save_freq != -1 and epoch % opt.save_freq == 0:
        #     print('saving random test results in {}'.format(opt.exprdir))
        #     save_random_tensors(opt, epoch, model, dataloader['test'])
        if 'bf3d' in model.model_names:
            for name, param in model.bf3d.named_parameters():
                print(name)
                print(param)
                if name.endswith("0.sigma_x"):
                    sigmax1_1 = param.item()
                elif name.endswith("0.sigma_y"):
                    sigmay1_1 = param.item()
                elif name.endswith("0.sigma_z"):
                    sigmaz1_1 = param.item()
                elif name.endswith("0.color_sigma"):
                    color1_1 = param.item()
                elif name.endswith("1.sigma_x"):
                    sigmax2_1 = param.item()
                elif name.endswith("1.sigma_y"):
                    sigmay2_1 = param.item()
                elif name.endswith("1.sigma_z"):
                    sigmaz2_1 = param.item()
                elif name.endswith("1.color_sigma"):
                    color2_1 = param.item()
                elif name.endswith("2.sigma_x"):
                    sigmax3_1 = param.item()
                elif name.endswith("2.sigma_y"):
                    sigmay3_1 = param.item()
                elif name.endswith("2.sigma_z"):
                    sigmaz3_1 = param.item()
                elif name.endswith("2.color_sigma"):
                    color3_1 = param.item()


                    
        if 'bf3d' in model.model_names:
            with open(opt.log_file, mode='a') as f:
                f.write("{},{:.8f},{:.8f},{:.8f},{:.8f},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    epoch, train_loss, train_psnr, valid_loss, valid_psnr, sigmax1_1, sigmay1_1, sigmaz1_1, color1_1, sigmax2_1, sigmay2_1, sigmaz2_1, color2_1, sigmax3_1, sigmay3_1, sigmaz3_1, color3_1
                ))
        else:
            with open(opt.log_file, mode='a') as f:
                f.write("{},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
                    epoch, train_loss, train_psnr, valid_loss, valid_psnr
                ))

        model.update_learning_rate(valid_loss)
        opt.epoch = epoch + 1




# import torch

# from utils.trainer import train_net
# from data import create_dataset
# from models import create_model

# from options.train_options import TrainOptions

# if __name__ == '__main__':
#     opt = TrainOptions().parse()
#     dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
#     dataset_size = len(dataloader['train'])    # get the number of images in the dataset.
    
#     print('\nThe number of training iteration = {}\n'.format(dataset_size))

    
#     model = create_model(opt)      # create a model given opt.model and other options
#     model.setup(opt)               # regular setup: load and print networks; create schedulers

#     best_psnr = 0
#     start_epoch = opt.epoch
#     if start_epoch == 1 and 'bf3d1' in model.model_names:
#         with open(opt.log_file, mode='w') as f:
#             f.write("epoch,train_loss,train_psnr,valid_loss,valid_psnr,sigmax1,sigmay1,sigmaz1,color1,sigmax2,sigmay2,sigmaz2,color2,sigmax3,sigmay3,sigmaz3,color3\n")
#     elif start_epoch == 1 and 'bf3d2' in model.model_names:
#         with open(opt.log_file, mode='w') as f:
#             f.write("epoch,train_loss,train_psnr,valid_loss,valid_psnr,sigmax1,sigmay1,sigmaz1,color1,sigmax2,sigmay2,sigmaz2,color2,sigmax3,sigmay3,sigmaz3,color3\n")
#     elif start_epoch == 1 and 'bf3d3' in model.model_names:
#         with open(opt.log_file, mode='w') as f:
#             f.write("epoch,train_loss,train_psnr,valid_loss,valid_psnr,sigmax1,sigmay1,sigmaz1,color1,sigmax2,sigmay2,sigmaz2,color2,sigmax3,sigmay3,sigmaz3,color3\n")
#     elif start_epoch == 1:
#         with open(opt.log_file, mode='w') as f:
#             f.write("epoch,train_loss,train_psnr,valid_loss,valid_psnr\n")


#     for epoch in range(start_epoch, opt.n_epochs+1):
#         train_loss, train_psnr = train_net(opt, model, dataloader['train'], is_train=True)
#         valid_loss, valid_psnr = train_net(opt, model, dataloader['validation'], is_train=False)

#         print('saving the latest model (epoch {}, total_iters {})'.format(epoch, opt.n_epochs))
#         model.remove_networks('lates')
#         model.save_networks('latest', epoch, valid_loss, valid_psnr)
#         # model.save_networks(epoch, epoch, valid_loss)
#         if opt.save_epoch_freq != -1 and epoch % opt.save_epoch_freq == 0:
#             print('saving the model (epoch {}, total_iters {}) with frequency {}'.format(epoch, opt.n_epochs, opt.save_epoch_freq))
#             model.save_networks(epoch, epoch, valid_loss, valid_psnr)

#         # if opt.save_freq != -1 and epoch % opt.save_freq == 0:
#         #     print('saving random test results in {}'.format(opt.exprdir))
#         #     save_random_tensors(opt, epoch, model, dataloader['test'])
#         if 'bf3d1' in model.model_names:
#             for name, param in model.bf3d1.named_parameters():
#                 if name.endswith("1.sigma_x"):
#                     sigmax1 = param
#                 elif name.endswith("1.sigma_y"):
#                     sigmay1 = param
#                 elif name.endswith("1.sigma_z"):
#                     sigmaz1 = param
#                 elif name.endswith("1.color_sigma"):
#                     color1 = param
#                 elif name.endswith("2.sigma_x"):
#                     sigmax2 = param
#                 elif name.endswith("2.sigma_y"):
#                     sigmay2 = param
#                 elif name.endswith("2.sigma_z"):
#                     sigmaz2 = param
#                 elif name.endswith("2.color_sigma"):
#                     color2 = param
#                 elif name.endswith("3.sigma_x"):
#                     sigmax3 = param
#                 elif name.endswith("3.sigma_y"):
#                     sigmay3 = param
#                 elif name.endswith("3.sigma_z"):
#                     sigmaz3 = param
#                 elif name.endswith("3.color_sigma"):
#                     color3 = param
#         elif 'bf3d' in model.model_names:
#             for name, param in model.bf3d.named_parameters():
#                 if name.endswith("1.sigma_x"):
#                     sigmax1 = param
#                 elif name.endswith("1.sigma_y"):
#                     sigmay1 = param
#                 elif name.endswith("1.sigma_z"):
#                     sigmaz1 = param
#                 elif name.endswith("1.color_sigma"):
#                     color1 = param
#                 elif name.endswith("2.sigma_x"):
#                     sigmax2 = param
#                 elif name.endswith("2.sigma_y"):
#                     sigmay2 = param
#                 elif name.endswith("2.sigma_z"):
#                     sigmaz2 = param
#                 elif name.endswith("2.color_sigma"):
#                     color2 = param
#                 elif name.endswith("3.sigma_x"):
#                     sigmax3 = param
#                 elif name.endswith("3.sigma_y"):
#                     sigmay3 = param
#                 elif name.endswith("3.sigma_z"):
#                     sigmaz3 = param
#                 elif name.endswith("3.color_sigma"):
#                     color3 = param

                    
#         if 'bf3d1' in model.model_names:
#             f.write("{},{:.8f},{:.8f},{:.8f},{:.8f},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
#                 epoch, train_loss, train_psnr, valid_loss, valid_psnr, sigmax1, sigmay1, sigmaz1, color1, sigmax2, sigmay2, sigmaz2, color2, sigmax3, sigmay3, sigmaz3, color3
#             ))
#         else:
#             with open(opt.log_file, mode='a') as f:
#                 f.write("{},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
#                     epoch, train_loss, train_psnr, valid_loss, valid_psnr
#                 ))

#         model.update_learning_rate(valid_loss)
#         opt.epoch = epoch + 1