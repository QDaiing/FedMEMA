import os
from datetime import datetime
import logging
import random
import time
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from options import args_parser
from models import model, hemis
from utils.lr_scheduler import LR_Scheduler
from utils.predict import AverageMeter, local_test, softmax_output_dice_class4, softmax_output_dice_class5
from utils import criterions
from dataset.data_utils import init_fn
from dataset.datasets import Brats_train, Brats_test

def test_softmax(test_loader, model, device, dataname = 'BRATS2020'):
    
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().to(device)    # .cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'WT-whole', 'TC-core', 'ET-enhancing', 'enhancing_postpro'
        class_separate = 'NET-ncr_net', 'ED-edema', 'ET-enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'WT-whole', 'TC-core', 'ET-enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
        

    for i, data in enumerate(test_loader):
        target = data[1].to(device)
        x = data[0].to(device)
        names = data[-1]
        # if feature_mask is not None:
        #     mask = torch.from_numpy(np.array(feature_mask))     # torch.Size([1, 4])
        #     mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        # else:
        #     mask = data[2]
        # mask = mask.cuda()
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int(np.ceil((H - 80) / (80 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int(80 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 80)

        w_cnt = np.int(np.ceil((W - 80) / (80 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int(80 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 80)

        z_cnt = np.int(np.ceil((Z - 80) / (80 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int(80 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 80)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().to(device) #.cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+80, w:w+80, z:z+80] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().to(device) #.cuda()
        # model.is_training=False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                    pred_part, _ = model(x_input)
                    pred[:, :, h:h+80, w:w+80, z:z+80] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)    # torch.Size([1, H, W, D])

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg +=  str(scores_evaluation[k].mean()) + ', '
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    logging.info(msg)
    model.train()
    return vals_evaluation.avg

if __name__ == '__main__':
    args = args_parser()
    
    args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    args.save_path = args.save_root + '/' + str(args.version) + '_%s'%(timestamp)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    args.modelfile_path = os.path.join(args.save_path, 'model_files')
    if not os.path.exists(args.modelfile_path):
        os.makedirs(args.modelfile_path)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        filename=args.save_path + '/cl_log.txt')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)
    
    writer = SummaryWriter(os.path.join(args.save_path, 'TBlog'))
    
    ##### modality missing mask
    # masks = [[True, True, True, True], [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True]]
    masks = [[True, True, True, True], [True, True, False,False], [False, False, True, True], 
             [True, False, True, False], [False, True, False, True], [True, False, False, True], [False, True, True, False]]
    # masks = [[True, True, True, True], [True, False, False,False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]
    
    masks_torch = torch.from_numpy(np.array(masks))
    # mask_name = ['flairt1cet1t2', 'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2']
    mask_name = ['flairt1cet1t2', 'flairt1ce', 't1t2', 'flairt1', 't1cet2', 'flairt2', 't1cet1']
    # mask_name = ['flairt1cet1t2', 'flair', 't1ce', 't1', 't2']
    print (masks_torch.int())

    ########## setting seed for deterministic
    if args.deterministic:
        # cudnn.enabled = False
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    ########## setting device and gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########## setting models
    args.num_class = 4   # for Brats2020
    model = model.E4D4Model(num_cls=args.num_class, is_lc=False)
    # model = hemis.Unet3D_HeMIS(basic_dims=16)
    model = model.to(args.device)

    ########## resume model from checkpoint
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    
    ########## setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    #modal_list = ['flair', 't1ce', 't1', 't2']
    logging.info(str(args))

    train_file = args.train_file[args.client_num]
    train_set = Brats_train(transforms=args.train_transforms, root=args.datapath,
                            modal=args.chose_modal, num_cls=args.num_class, train_file=train_file)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=8,
        pin_memory=True, shuffle=True, worker_init_fn=init_fn)
    
    valid_set = Brats_test(transforms=args.test_transforms, root=args.datapath,
                           modal=args.chose_modal, test_file=args.valid_file)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    test_set = Brats_test(transforms=args.test_transforms, root=args.datapath,
                           modal=args.chose_modal, test_file=args.test_file)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    logging.info('the Brats dataset with modal {}'.format(mask_name[args.client_num]))
    logging.info('the length of Brats train dataset is {}'.format(len(train_set)))
    logging.info('the length of Brats valid dataset is {}'.format(len(valid_set)))
    logging.info('the length of Brats test dataset is {}'.format(len(test_set)))

    mask = masks_torch[args.client_num]
    ########## Training ##########
    start = time.time()
    best_dice = 0.0
    logging.info('-'*10+' start CL training '+'-'*10)
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        e = time.time()
        model.train()
        for i, data in enumerate(train_loader):
            step = (i+1) + epoch * len(train_loader)
            
            vol_batch, msk_batch = data[0].to(args.device), data[1].to(args.device)
            # vol_batch - torch.Size([4, 1, 80, 80, 80])
            # msk_batch - torch.Size([4, 4, 80, 80, 80])
            names = data[-1]

            msk = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
            # msk = mask

            msk = msk.to(args.device)
            model.is_training = True
            
            # fuse_pred = model(vol_batch.transpose(1,0).unsqueeze(2), msk)
            fuse_pred, prm_preds, _, sep_preds = model(vol_batch, msk, None, None, None, None)
            # pred - torch.Size([1, 4, 80, 80, 80])
            
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            prm_cross_loss = torch.zeros(1).float().to(args.device)
            prm_dice_loss = torch.zeros(1).float().to(args.device)
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, msk_batch, num_cls=args.num_class)
                prm_dice_loss += criterions.dice_loss(prm_pred, msk_batch, num_cls=args.num_class)
            prm_loss = prm_cross_loss + prm_dice_loss

            sep_cross_loss = torch.zeros(1).float().to(args.device)
            sep_dice_loss = torch.zeros(1).float().to(args.device)
            for pi in range(sep_preds.shape[0]):
                sep_pred = sep_preds[pi]
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, msk_batch, num_cls=args.num_class)
                sep_dice_loss += criterions.dice_loss(sep_pred, msk_batch, num_cls=args.num_class)
            sep_loss = sep_cross_loss + sep_dice_loss

            loss = fuse_loss + prm_loss + sep_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ##### log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_loss', fuse_loss.item(), global_step=step)
            writer.add_scalar('prm_loss', prm_loss.item(), global_step=step)
            writer.add_scalar('sep_loss', sep_loss.item(), global_step=step)

            # 训练过程结果可视化
            if args.visualize and (i%30==0):
                image = vol_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)  # torch.Size([5, 3, 80, 80])
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Tcrop_Image', grid_image, i)

                image = fuse_pred[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)  # torch.Size([5, 3, 80, 80])
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_1',
                                 grid_image, i)

                image = msk_batch[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)  # torch.Size([5, 3, 80, 80])
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label_1',
                                 grid_image, i)

                image = fuse_pred[0, 2:3, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)  # torch.Size([5, 3, 80, 80])
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_2',
                                 grid_image, i)

                image = msk_batch[0, 2:3, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)  # torch.Size([5, 3, 80, 80])
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label_2',
                                 grid_image, i)
                
                image = fuse_pred[0, 3:4, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)  # torch.Size([5, 3, 80, 80])
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_3',
                                 grid_image, i)

                image = msk_batch[0, 3:4, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)  # torch.Size([5, 3, 80, 80])
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label_3',
                                 grid_image, i)

            if args.verbose and (i%10==0):
                msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), len(train_loader), loss.item())
                msg += 'loss_fuse:{:.4f}, loss_prm:{:.4f}, loss_sep:{:.4f}'.format(fuse_loss.item(), prm_loss.item(), sep_loss.item())
                logging.info(msg)
        
        logging.info('train cost time per epoch: {}'.format(time.time()-e))

        glbFs = { }
        ##### Evaluate the model in specified interval
        if (epoch+1) % 20 ==0 and (epoch+1) > 400:
            with torch.no_grad():
                logging.info(' ########## start eval in epoch {} ########## '.format(epoch+1))
                dice_score = local_test(args, valid_loader, model, args.device, args.dataname, glbFs, mask)
                avg_dice = sum(dice_score)/len(dice_score)
                logging.info('--- Eval at Epoch_{}, Avg_Scores: {:.4f}, cls_Dice: {}'.format((epoch+1), avg_dice*100, dice_score))
            if best_dice < avg_dice:
                best_dice = avg_dice
                file_name = os.path.join(args.modelfile_path, 'model_e{}_{:.4f}.pth'.format(epoch+1, best_dice))
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict()
                }, file_name
                )
                # ### test the model
                # with torch.no_grad():
                #     logging.info(' ########## after eval, test the model ########## ')
                #     test_dice_score = local_test(args, test_loader, model, args.device, args.dataname, glbFs, mask)
                #     test_avg_dice = sum(test_dice_score)/len(test_dice_score)
                #     logging.info('--- Test after Eval, Avg_Scores: {:.4f}, cls_Dice: {}'.format(test_avg_dice*100, test_dice_score))

    msg = 'Training Total Time for {} epochs: {:.4f} hours'.format(args.num_epochs, (time.time()-start)/3600)
    logging.info(msg)
    
    ### save the latest model 
    file_name = os.path.join(args.modelfile_path, 'model_last.pth')
    torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
                }, file_name
                )
    
    writer.close()

    