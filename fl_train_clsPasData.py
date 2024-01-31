from asyncore import write
from subprocess import check_output
from tabnanny import check
import torch
import os
import random
import numpy as np
import time
import copy
from torch import nn
from tqdm import tqdm
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.data_utils import init_fn

from models import model
from utils.fl_utils import EMA_cls_Fs, avg_EW, cluster_Fs, getClsFeatures, getClsPrototypes, getClusDict, getPrototype
from utils.lr_scheduler import LR_Scheduler
from utils import criterions
from dataset.datasets import Brats_test, Brats_train, GLB_Brats_train
from options import args_parser
from utils.predict import global_test, local_test, test_softmax

def local_training(args, device, msk, dataloader, model, client_idx, global_Fs, global_round):
    # set mode to train model
    model.train()
    model = model.to(device)
    start = time.time()
    epoch_loss = {'total':[], 'fuse':[], 'prm':[], 'sep':[]}

    # Set Optimizer for the local model update
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    step_lr = lr_schedule(optimizer, round)
    writer.add_scalar('lr_lc', step_lr, global_step=round)
    logging.info('############# client_{} local training ############'.format(client_idx+1))
    
    for iter in range(args.local_ep):
        batch_loss = {'total':[], 'fuse':[], 'prm':[], 'sep':[]}
        
        # step = epoch*len(dataloader) + len(dataloader)*round*args.local_ep
        for batch_idx, data in enumerate(dataloader):

            vol_batch, msk_batch = data[0].to(device), data[1].to(device)
            names = data[-1]
            # vol_batch - torch.Size([1, 1, 80, 80, 80]) # msk_batch - torch.Size([1, 4, 80, 80, 80])
            
            msk = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
            msk = msk.to(device)
            model.is_training = True

            Px1, Px2, Px3, Px4 = global_Fs['x1'].to(device), global_Fs['x2'].to(device), global_Fs['x3'].to(device), global_Fs['x4'].to(device)
            Px1, Px2 = Px1.reshape(-1, Px1.shape[-1]), Px2.reshape(-1, Px2.shape[-1])
            Px3, Px4 = Px3.reshape(-1, Px3.shape[-1]), Px4.reshape(-1, Px4.shape[-1])   # torch.Size([40, C])

            fuse_pred, prm_preds, _, sep_preds = model(vol_batch, msk, Px1, Px2, Px3, Px4)
            # pred - torch.Size([1, 4, 80, 80, 80])

            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            prm_cross_loss = torch.zeros(1).float().to(device)
            prm_dice_loss = torch.zeros(1).float().to(device)
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, msk_batch, num_cls=args.num_class)
                prm_dice_loss += criterions.dice_loss(prm_pred, msk_batch, num_cls=args.num_class)
            prm_loss = prm_cross_loss + prm_dice_loss

            sep_cross_loss = torch.zeros(1).float().to(device)
            sep_dice_loss = torch.zeros(1).float().to(device)
            for pi in range(sep_preds.shape[0]):
                sep_pred = sep_preds[pi]
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, msk_batch, num_cls=args.num_class)
                sep_dice_loss += criterions.dice_loss(sep_pred, msk_batch, num_cls=args.num_class)
            sep_loss = sep_cross_loss + sep_dice_loss

            loss = fuse_loss + prm_loss + sep_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.verbose and (batch_idx%10==0):
                logging.info('| Global Round : {} | Client: {} | Local Epoch : {} | [{}/{} ({:.0f}%)] Loss: {:.3f} | FuseLoss: {:.3f} | PrmLoss : {:.3f} | sep_loss : {:.3f}'
                      .format(global_round, client_idx+1, iter+1, batch_idx * len(data[0]),
                        len(dataloader.dataset),
                        100. * batch_idx / len(dataloader),
                        loss.item(), fuse_loss.item(), prm_loss.item(), sep_loss.item()))

            batch_loss['total'].append(loss.item())
            batch_loss['fuse'].append(fuse_loss.item())
            batch_loss['prm'].append(prm_loss.item())
            batch_loss['sep'].append(sep_loss.item())
            # torch.cuda.empty_cache()
            
        epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
        epoch_loss['fuse'].append(sum(batch_loss['fuse'])/len(batch_loss['fuse']))
        epoch_loss['prm'].append(sum(batch_loss['prm'])/len(batch_loss['prm']))
        epoch_loss['sep'].append(sum(batch_loss['sep'])/len(batch_loss['sep']))
        
    epoch_loss['total'] = sum(epoch_loss['total'])/len(epoch_loss['total'])
    epoch_loss['fuse'] = sum(epoch_loss['fuse'])/len(epoch_loss['fuse'])
    epoch_loss['prm'] = sum(epoch_loss['prm'])/len(epoch_loss['prm'])
    epoch_loss['sep'] = sum(epoch_loss['sep'])/len(epoch_loss['sep'])
    
    msg = 'client_{} local training total time: {:.4f} hours'.format(client_i+1, (time.time() - start)/3600)
    logging.info(msg)

    return [model.c1_encoder.state_dict(), model.c2_encoder.state_dict(), model.c3_encoder.state_dict(),
            model.c4_encoder.state_dict()], epoch_loss

def global_training(args, device, dataloader, model, modal_protos, round):

    model.train()
    model = model.to(device)
    start = time.time()

    Xscale_list = ['x1', 'x2', 'x3', 'x4']
    glb_Fs = {'x1':[], 'x2':[], 'x3':[], 'x4':[]}
    glb_Pnames = []

    # Set Optimizer for the global model update
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    step_lr = lr_schedule(optimizer, round)
    writer.add_scalar('lr_glb', step_lr, global_step=round)
    logging.info('############# global training on the server ############')
    
    for epoch in range(args.global_ep):
        
        e = time.time()
        step = epoch*len(dataloader) + len(dataloader)*round*args.global_ep
        
        for iter, data in enumerate(dataloader):
            # step = step+1
            vol, target, msk, p_name = data
            glb_Pnames += p_name
            vol_batch, msk_batch = vol.to(device), target.to(device)
            mask = msk.to(device)   # tensor([[True, True, True, True]], device='cuda:0')
            # vol_batch - torch.Size([B, 4, 80, 80, 80])
            # msk_batch - torch.Size([B, 4, 80, 80, 80])
            # mask - torch.Size([B, 4])
            model.is_training = True
            fuse_pred, prm_preds, features, sep_preds = model(vol_batch, mask, None,None,None,None) 
            # fuse_pred - torch.Size([1, 4, 80, 80, 80])
            # sep_preds - 4 * torch.Size([1, 4, 80, 80, 80])
            # prm_preds - 4 * torch.Size([1, 4, 80, 80, 80])
                    
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, msk_batch, num_cls=args.num_class)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            prm_cross_loss = torch.zeros(1).float().to(device)
            prm_dice_loss = torch.zeros(1).float().to(device)
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, msk_batch, num_cls=args.num_class)
                prm_dice_loss += criterions.dice_loss(prm_pred, msk_batch, num_cls=args.num_class)
            prm_loss = prm_cross_loss + prm_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for pi in range(sep_preds.shape[0]):
                sep_pred = sep_preds[pi]
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, msk_batch, num_cls=args.num_class)
                sep_dice_loss += criterions.dice_loss(sep_pred, msk_batch, num_cls=args.num_class)
            sep_loss = sep_cross_loss + sep_dice_loss

            for i in range(len(features)):
                scale = Xscale_list[i]
                # 对应当前尺度下的融合模态特征图
                fusion_features = features[i]       # Fx4 - torch.Size([1, 128, 10, 10, 10])
                cls_F = getClsPrototypes(fusion_features, msk_batch)  # (cls, C)
                glb_Fs[scale]  += cls_F

            loss = fuse_loss + prm_loss + sep_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('GlobalTrain/loss', loss.item(), global_step=step)
            # writer.add_scalar('GlobalTrain/proto_align_loss', proto_align_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('GlobalTrain/prm_dice_loss', prm_dice_loss.item(), global_step=step)

            if args.verbose and (iter%10==0):
                msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.local_ep, (iter), len(dataloader), loss.item())
                msg += 'fusecross:{:.4f}, fusedice:{:.4f}, '.format(fuse_cross_loss.item(), fuse_dice_loss.item())
                msg += 'sepcross:{:.4f}, sepdice:{:.4f}, '.format(sep_cross_loss.item(), sep_dice_loss.item())
                msg += 'prmcross:{:.4f}, prmdice:{:.4f}'.format(prm_cross_loss.item(), prm_dice_loss.item())
                # msg += 'ProtoAlignLoss:{:.4f}'.format(proto_align_loss.item())
                logging.info(msg)

    msg = 'server global training total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    return model.state_dict(), glb_Fs, glb_Pnames

def uploadLCweightsandGLBupdate(masks, server_model, local_weights, train_loader, protos, round):
        lc1_mask, lc2_mask, lc3_mask, lc4_mask = masks[0], masks[1], masks[2], masks[3]
        modals_E = []
        for i in range(len(lc1_mask)):  # 针对每一个模态的特异编码器
            c_E = avg_EW(local_weights[0][i], local_weights[1][i], local_weights[2][i], local_weights[3][i], 
                        lc1_mask[i], lc2_mask[i], lc3_mask[i], lc4_mask[i])
            modals_E.append(c_E)

        server_model.c1_encoder.load_state_dict(modals_E[0])
        server_model.c2_encoder.load_state_dict(modals_E[1])
        server_model.c3_encoder.load_state_dict(modals_E[2])
        server_model.c4_encoder.load_state_dict(modals_E[3])
        logging.info('-'*20+'the Global Server has received client weights'+'-'*20)
        
        ### global training
        glb_w, glb_protos, glb_Pnames = global_training(args, args.device, glb_trainloader, server_model, None, round)   

        return glb_w, glb_protos, glb_Pnames

def downloadGLBweights(glb_w, model_clients):
    c1_w = {k.replace('c1_encoder.',''): v.cpu() for k,v in glb_w.items() if 'c1_encoder' in k}
    c2_w = {k.replace('c2_encoder.',''): v.cpu() for k,v in glb_w.items() if 'c2_encoder' in k}
    c3_w = {k.replace('c3_encoder.',''): v.cpu() for k,v in glb_w.items() if 'c3_encoder' in k}
    c4_w = {k.replace('c4_encoder.',''): v.cpu() for k,v in glb_w.items() if 'c4_encoder' in k}

    for i in range(len(model_clients)): # 简单起见，直接全部加载，但训练时其实只用到了存在模态编码器
        model_clients[i].c1_encoder.load_state_dict(c1_w)    # flair模态 
        model_clients[i].c2_encoder.load_state_dict(c2_w)    # t1ce模态
        model_clients[i].c3_encoder.load_state_dict(c3_w)    # t1模态
        model_clients[i].c4_encoder.load_state_dict(c4_w)    # t2模态

    logging.info('-'*20+' the Global Server send the glb_Weights to clients ' + '-'*20)

if __name__ == '__main__':
    ### global model - 4 模态特异Encoder & 1 模态融合Decoder
    ### local model - 模态特异Encoder & Decoder
    ### FL过程中只传递Encoder参数，不做其他处理
    args = args_parser()
    
    # 数据预处理遵循RFNet
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
                        filename=args.save_path + '/fl_log.txt')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)
    
    writer = SummaryWriter(os.path.join(args.save_path, 'TBlog'))
    
    ##### modality missing mask
    # masks = [[True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True]]
    # masks = [[True, True, False,False], [False, False, True, True], [True, False, True, False], [False, True, False, True]]
    masks = [[True, False, False,False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]
    masks_torch = torch.from_numpy(np.array(masks))
    # mask_name = ['flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2']
    # mask_name = ['flairt1ce', 't1t2', 'flairt1', 't1cet2']
    mask_name = ['flair', 't1ce', 't1', 't2']
    logging.info(masks_torch.int())

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
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.local_devices = []
    for i in range(args.client_num):
        args.local_devices.append(torch.device('cuda:{}'.format(i+1)))
    
    ########## setting global and local model
    server_model = model.E4D4Model(num_cls=args.num_class, is_lc=False)
    client_model = model.E4D4Model(num_cls=args.num_class, is_lc=True)

    lr_schedule = LR_Scheduler(args.lr, args.c_rounds)
    ########## FL setting ##########
    # define dataset, model, optimizer for each clients 
    dataloader_clients, validloader_clients, testloader_clients = [], [], []
    model_clients = []
    optimizer_clients = []
    client_counts, client_weights = [], []     ### FedAvg Setting
    args.train_file = {'glb':"/disk3/qd/edFL/data/BRATS2020_Training_none_npy/split_0/glb_train.txt", 
                1:"/disk3/qd/edFL/data/BRATS2020_Training_none_npy/split_0/c1_train.txt", 
                2:"/disk3/qd/edFL/data/BRATS2020_Training_none_npy/split_0/c2_train.txt", 
                3:"/disk3/qd/edFL/data/BRATS2020_Training_none_npy/split_0/c3_train.txt", 
                4:"/disk3/qd/edFL/data/BRATS2020_Training_none_npy/split_0/c4_train.txt"}
    args.valid_file = "/disk3/qd/edFL/data/BRATS2020_Training_none_npy/split_0/val.txt"
    args.test_file = "/disk3/qd/edFL/data/BRATS2020_Training_none_npy/split_0/test.txt"
    modal_list = ['flair', 't1ce', 't1', 't2']
    logging.info(str(args))

    for client_idx in range(args.client_num):
        chose_modal = 'all'
        lc_train_file = args.train_file[client_idx+1]
        data_set = Brats_train(transforms=args.train_transforms, root=args.datapath, 
                                modal=chose_modal, num_cls=args.num_class, train_file=lc_train_file)
        data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size, num_workers=8,
                                pin_memory=True, shuffle=True, worker_init_fn=init_fn)
        valid_set = Brats_test(transforms=args.test_transforms, root=args.datapath, 
                                modal=chose_modal, test_file=args.valid_file)
        valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        test_set = Brats_test(transforms=args.test_transforms, root=args.datapath, 
                                modal=chose_modal, test_file=args.test_file)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        dataloader_clients.append(data_loader)
        validloader_clients.append(valid_loader)
        testloader_clients.append(test_loader)
        logging.info('Client-{} : Brats dataset with modal {}'.format(client_idx+1, mask_name[client_idx]))
        logging.info('the length of Brats dataset is {} : {} : {}'.format(len(data_set), len(valid_set), len(test_set)))

        device = args.local_devices[client_idx]

        net = copy.deepcopy(client_model)   # .to(device)  # .to(args.device)
        model_clients.append(net) 
    
    # ##### resume
    # cpl = torch.load("/disk3/qd/FedMEMA/results/fltrain_AvgE_1modal_clusPasData_05011532/model_files/sever_model_best.pth")
    # server_model.load_state_dict(cpl['state_dict'])
    # cpl1 = torch.load("/disk3/qd/FedMEMA/results/fltrain_AvgE_1modal_clusPasData_05011532/model_files/client-1_round_920_model_best.pth")
    # # model_clients[0].load_state_dict(cpl1['state_dict'])
    # cpl2 = torch.load("/disk3/qd/FedMEMA/results/fltrain_AvgE_1modal_clusPasData_05011532/model_files/client-2_round_940_model_best.pth")
    # # model_clients[1].load_state_dict(cpl2['state_dict'])
    # cpl3 = torch.load("/disk3/qd/FedMEMA/results/fltrain_AvgE_1modal_clusPasData_05011532/model_files/client-3_round_940_model_best.pth")
    # # model_clients[2].load_state_dict(cpl3['state_dict'])
    # cpl4 = torch.load("/disk3/qd/FedMEMA/results/fltrain_AvgE_1modal_clusPasData_05011532/model_files/client-4_round_900_model_best.pth")
    # # model_clients[3].load_state_dict(cpl4['state_dict'])
    # checkpoint = [cpl1['state_dict'], cpl2['state_dict'], cpl3['state_dict'], cpl4['state_dict']]

    ##### global dataset 
    glb_train_file = args.train_file['glb']
    glb_dataset = GLB_Brats_train(transforms=args.train_transforms, root=args.datapath, 
                                modal='all', num_cls=args.num_class, train_file=glb_train_file)
    glb_trainloader = DataLoader(dataset=glb_dataset, batch_size=1, num_workers=8,
                                pin_memory=True, shuffle=True, worker_init_fn=init_fn)
    glb_validset = Brats_test(transforms=args.test_transforms, root=args.datapath, 
                                modal='all', test_file=args.valid_file)
    glb_validloader = DataLoader(dataset=glb_validset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    glb_testset = Brats_test(transforms=args.test_transforms, root=args.datapath, 
                                modal='all', test_file=args.test_file)
    glb_testloader = DataLoader(dataset=glb_testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    logging.info('Global : Brats dataset with all modal')
    logging.info('the length of Brats dataset is {} : {} : {}'.format(len(glb_dataset), len(glb_validset), len(glb_testset)))

    best_dices = [0.0, 0.0, 0.0, 0.0]
    best_dice = 0.0
    global_Fs = { } # 'x1': [], 'x2':[], 'x3':[], 'x4':[]
    Xscale_list = ['x1', 'x2', 'x3', 'x4']
    

    
    ########## FL Training ##########
    for round in tqdm(range(args.c_rounds+1)):

        start = time.time()
        if round==0:
            ##### global training
            glb_w, glb_protos, glb_Pnames = global_training(args, args.device, glb_trainloader, server_model, None, round)
            ## 按某一尺度特征 按类别进行聚类
            cls_glb_clusDict = {}
            for scale in Xscale_list:
                clu_Fs, labels = cluster_Fs(glb_protos[scale], asCls=True)  # [k, cls*C]
                clu_Fs = np.stack(clu_Fs, axis=1)   # 再按类别cat起来 [k, cls, C]
                clu_Fs = torch.from_numpy(clu_Fs)
                global_Fs[scale] = clu_Fs
                if scale=='x4':
                    for c in range(args.num_class):
                        glb_clusDict = getClusDict(glb_Pnames, labels[c])
                        cls_glb_clusDict[c] = glb_clusDict
        else: 
            ##### local training
            local_weights, local_losses, local_protos = [], [], {}
            logging.info(f'\n | Global Training Round : {round} |')
            for client_i in range(args.client_num):
                dataloader_c = dataloader_clients[client_i]
                device = args.local_devices[client_i]
                model_c = model_clients[client_i]
                # if round ==1:
                #     model_c.load_state_dict(checkpoint[client_i])
                mask = masks_torch[client_i]
                w, loss = local_training(args, device, mask, dataloader_c, model_c, client_i, global_Fs, round)
                
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss['total']))
                # local_protos[client_i] = agg_protos
                writer.add_scalar('LocalTrain/total_Loss/client_' + str(client_i + 1), loss['total'], round)
                writer.add_scalar('LocalTrain/Loss_fuse/client_' + str(client_i + 1), loss['fuse'], round)
                writer.add_scalar('LocalTrain/Loss_prm/client_' + str(client_i + 1), loss['prm'], round)
                writer.add_scalar('LocalTrain/Loss_sep/client_' + str(client_i + 1), loss['sep'], round)

            # global Aggre and Fusion
            # local_protos - 4个客户端即4个模态的4个类别原型
            glb_w, glb_protos, glb_Pnames = uploadLCweightsandGLBupdate(masks_torch, server_model, local_weights, glb_trainloader, None, round)   
            for scale in Xscale_list:
                scale_protos = torch.stack(glb_protos[scale], dim=0)    # (len, cls, C)
                cls_protos = []
                for c in range(args.num_class):
                    clu_Fs = EMA_cls_Fs(global_Fs[scale][:,c], scale_protos[:,c], glb_Pnames, cls_glb_clusDict[c], round)
                    cls_protos.append(clu_Fs)
                cls_protos = np.stack(cls_protos, axis=1)   # (k, cls, C)
                global_Fs[scale] = (torch.from_numpy(cls_protos)).float()

        ##### Eval the model after aggregation and 10 round
        if round%2 == 0 and round>1:
            logging.info('-'*20 + 'Test All the Models per 10 round'+ '-'*20)
            with torch.no_grad():
                # test clients
                for c in range(args.client_num):
                    c_validloader = validloader_clients[c]
                    c_model = model_clients[c]
                    device = args.local_devices[c]
                    msk = masks[c]
                    print(c_model.decoder_fuse.d3_c1.conv.weight[10,10,1,1], "hhhh", msk)
                    dice_score = local_test(args, c_validloader, c_model, device, 'BRATS2020', global_Fs, msk)
                    avgdice_score = sum(dice_score)/len(dice_score)
                    logging.info('--- Eval at round_{}, Avg_Scores: {:.4f}, cls_Dice: {}'
                                                        .format((round), avgdice_score*100, dice_score))
                    writer.add_scalar('Eval_AvgDice/client_'+str(c+1), avgdice_score*100, round)
                    
                    if best_dices[c] < avgdice_score:
                        best_dices[c] = avgdice_score
                        torch.save({
                            'round': round+1,
                            'dice': dice_score,
                            'state_dict': c_model.state_dict()
                        }, args.modelfile_path + '/client-%d_round_%d_model_best.pth'%(c+1, round))
                        
                        # ### # 同时test一下模型
                        # logging.info('-'*15+' Test all the Model '+'-'*15)
                        # c_testloader = testloader_clients[c]
                        # test_dice = test_softmax(args, c_testloader, c_model, device, 'BRATS2020', global_Fs)
                        # logging.info('client {} test AvgDice : {:.4f}, cls_Dice: {}'.format(c+1, 
                        #                 (sum(test_dice)/len(test_dice))*100, test_dice))
                        # writer.add_scalar('Test_AvgDice/client_'+str(c+1), (sum(test_dice)/len(test_dice))*100, round)
                        # ### # ###########

                # test server
                logging.info('-'*15+' Test the Global Model '+'-'*15)
                mask = [True, True, True, True]
                glbdice = global_test(glb_validloader, server_model, args.device, 'BRATS2020', mask)
                avg_glbdice = sum(glbdice)/len(glbdice)
                logging.info('--- Eval at round_{}, Avg_Scores: {:.4f}, cls_Dice: {}'
                                .format((round), avg_glbdice*100, glbdice))
                writer.add_scalar('Eval_AvgDice/server', avg_glbdice, round)

                if best_dice < avg_glbdice:
                    best_dice = avg_glbdice
                    torch.save({
                        'round': round+1,
                        'dice': glbdice,
                        'state_dict': server_model.state_dict()
                    }, args.modelfile_path + '/sever_model_best.pth')

                    # ### # 同时test一下模型
                    # logging.info('-'*15+' Test the Global Model '+'-'*15)
                    # test_glbdice = global_test(glb_testloader, server_model, args.device, 'BRATS2020', mask)
                    # logging.info('global test AvgDice : {:.4f}, cls_Dice: {}'.format( 
                    #                 (sum(test_glbdice)/len(test_glbdice))*100, test_glbdice))
                    # writer.add_scalar('Test_AvgDice/server', (sum(test_glbdice)/len(test_glbdice))*100, round)
                    # ### # ##########
        
        downloadGLBweights(glb_w, model_clients)
        logging.info('*'*10+'FL train a round total time: {:.4f} hours'.format((time.time() - start)/3600)+'*'*10)
        
        for c in range(args.client_num):
            print(model_clients[c].decoder_fuse.d3_c1.conv.weight[10,10,1,1])
        
        import json
        with open(args.save_path+"/glb_Pdict.json", 'w') as f1:
            f1.write(json.dumps(cls_glb_clusDict, indent=4, ensure_ascii=False))
        if round != 0:     
            for scale in Xscale_list:
                Fs = global_Fs[scale].numpy()
                np.save(args.save_path+'/glb_'+str(scale)+'.npy', Fs)

    writer.close()    