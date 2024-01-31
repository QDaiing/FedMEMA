import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import nibabel as nib
import scipy.misc

cudnn.benchmark = True

path = os.path.dirname(__file__)


def softmax_output_dice_class4(output, target):
    eps = 1e-8
    ### NET
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    ### ED
    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    # ET - the enhancing tumor
    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    # if torch.sum(o3) < 500:
    #    o4 = o3 * 0.0
    # else:
    #    o4 = o3
    # t4 = t3
    # intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    # denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    # enhancing_dice_postpro = intersect4 / denominator4

    # WT - the whole tumor
    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    # TC - the tumor core
    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1,2,3)) + eps
    denominator5 = torch.sum(o5, dim=(1,2,3)) + torch.sum(t5, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(non_enhancing_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def test_softmax(args, test_loader, model, device, dataname, glb_features):
    
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().to(device)    # .cuda()
    glb_protos = []
    if glb_features==None:
            logging.info('*'*10+'the local test without Global Features!'+'*'*10)
    else:        
        ##### 聚类特征图
        Px1, Px2, Px3, Px4 = glb_features['x1'].to(device), glb_features['x2'].to(device), glb_features['x3'].to(device), glb_features['x4'].to(device)
        Px1, Px2 = Px1.reshape(-1, Px1.shape[-1]), Px2.reshape(-1, Px2.shape[-1]), 
        Px3, Px4 = Px3.reshape(-1, Px3.shape[-1]), Px4.reshape(-1, Px4.shape[-1]) 

        logging.info('*'*10+'the local test with Global Features!'+'*'*10)

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
        model.is_training = True # False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                    # 1.传递GLb_Features
                    pred_part, _ = model(x_input, Px1, Px2, Px3, Px4)   # glb_protos
                    # 2.不传递
                    # pred_part, _ = model(x_input)
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
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            # logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print(msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg

def local_test(args, test_loader, model, device, dataname, glb_features, modal_mask, writer = None):
    model = model.to(device)
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().to(device)    # .cuda()
    glb_protos = []
    if glb_features== {}:
            Px1, Px2, Px3, Px4 = None, None, None, None
            logging.info('*'*10+'the local test without Global Features!'+'*'*10)
    else:        
        ##### 聚类特征图
        Px1, Px2, Px3, Px4 = glb_features['x1'].to(device), glb_features['x2'].to(device), glb_features['x3'].to(device), glb_features['x4'].to(device)
        Px1, Px2 = Px1.reshape(-1, Px1.shape[-1]), Px2.reshape(-1, Px2.shape[-1]), 
        Px3, Px4 = Px3.reshape(-1, Px3.shape[-1]), Px4.reshape(-1, Px4.shape[-1]) 

        logging.info('*'*10+'the local test with Global Features!'+'*'*10)

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
        if modal_mask is not None:
            mask = torch.from_numpy(np.array(modal_mask)).unsqueeze(0)     # torch.Size([1, 4])
            # mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.to(device)
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
        model.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    with torch.no_grad():
                        x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                        # 1.传递GLb_Features, RFNet
                        model.is_gen = False
                        pred_part, _, _ = model(x_input, mask, Px1, Px2, Px3, Px4)   # glb_protos
                        # 不传递
                        # pred_part, _ = model(x_input)
                        # 2. HeMIS
                        #pred_part, _ = model(x_input.transpose(1,0).unsqueeze(2), mask, Px1, Px2, Px3, Px4)
                        # 3. mmmodel
                        # pred_part, _, _, _ = model(x_input, mask, Px1, Px2, Px3, Px4)
                        pred[:, :, h:h+80, w:w+80, z:z+80] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)    # torch.Size([1, H, W, D])

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            if writer is not None:
                writer.writerow([names[0], scores_evaluation[0,0], scores_evaluation[0,1],scores_evaluation[0,2]])
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print(msg)
    # logging.info(msg)
    model.train()
    model = model.cpu()
    return vals_evaluation.avg

def test_softmax_mED(args, test_loader, mask, model, device, dataname, glb_features):
    
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().to(device)    # .cuda()
    glb_protos = []
    if glb_features==None:
            logging.info('*'*10+'the local test without Global Features!'+'*'*10)
    else:        
        ##### 聚类特征图
        Px1, Px2, Px3, Px4 = glb_features['x1'].to(device), glb_features['x2'].to(device), glb_features['x3'].to(device), glb_features['x4'].to(device)
        Px1, Px2 = Px1.reshape(-1, Px1.shape[-1]), Px2.reshape(-1, Px2.shape[-1]), 
        Px3, Px4 = Px3.reshape(-1, Px3.shape[-1]), Px4.reshape(-1, Px4.shape[-1]) 

        logging.info('*'*10+'the local test with Global Features!'+'*'*10)

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
        pred1 = torch.zeros(len(names), num_cls, H, W, Z).float().to(device) #.cuda()
        pred2 = torch.zeros(len(names), num_cls, H, W, Z).float().to(device)
        pred3 = torch.zeros(len(names), num_cls, H, W, Z).float().to(device)
        model.is_training = True # False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                    # 1.传递GLb_Features
                    pred_part = model(x_input, mask, Px1, Px2, Px3, Px4)   # glb_protos
                    # 2.不传递
                    # pred_part, _ = model(x_input)
                    pred1[:, :, h:h+80, w:w+80, z:z+80] += pred_part[0]
                    pred2[:, :, h:h+80, w:w+80, z:z+80] += pred_part[1]
                    pred3[:, :, h:h+80, w:w+80, z:z+80] += pred_part[2]
        
        pred1 = pred1 / weight
        b = time.time()
        pred1 = pred1[:, :, :H, :W, :T]
        pred1 = torch.argmax(pred1, dim=1)    # torch.Size([1, H, W, D])
        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate_1, scores_evaluation_1 = softmax_output_dice_class4(pred1, target)
            if writer is not None:
                writer.writerow([names[0], scores_evaluation[0,0], scores_evaluation[0,1],scores_evaluation[0,2]])
        elif dataname == 'BRATS2015':
            scores_separate_1, scores_evaluation_1 = softmax_output_dice_class5(pred1, target)

        pred2 = pred2 / weight
        b = time.time()
        pred2 = pred2[:, :, :H, :W, :T]
        pred2 = torch.argmax(pred2, dim=1)    # torch.Size([1, H, W, D])
        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate_2, scores_evaluation_2 = softmax_output_dice_class4(pred2, target)
        elif dataname == 'BRATS2015':
            scores_separate_2, scores_evaluation_2 = softmax_output_dice_class5(pred2, target)
        
        pred3 = pred3 / weight
        b = time.time()
        pred3 = pred3[:, :, :H, :W, :T]
        pred3 = torch.argmax(pred3, dim=1)    # torch.Size([1, H, W, D])
        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate_3, scores_evaluation_3 = softmax_output_dice_class4(pred3, target)
        elif dataname == 'BRATS2015':
            scores_separate_3, scores_evaluation_3 = softmax_output_dice_class5(pred3, target)

        logging.info(mask)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)
            vals_separate.update((scores_separate_1[k]+scores_separate_2[k]+scores_separate_3[k])/3)
            vals_evaluation.update((scores_evaluation_1[k]+scores_evaluation_2[k]+scores_evaluation_3[k])/3)
            msg += ', '.join(['{}: {:.4f}, {:.4f}, {:.4f}'.format(k, v1,v2,v3) for k, v1,v2,v3 in zip(class_evaluation, scores_evaluation_1[k], scores_evaluation_2[k], scores_evaluation_3[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print(msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg


def global_test(test_loader, model, device, dataname = 'BRATS2020',
        feature_mask=None, mask_name=None):

    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().to(device)   # .cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
        
    for i, data in enumerate(test_loader):
        target = data[1].to(device)  #.cuda()
        x = data[0].to(device)       # .cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask)).unsqueeze(0)
            # mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.to(device)     #.cuda()
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
        weight1 = torch.zeros(1, 1, H, W, Z).float().to(device)    # .cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+80, w:w+80, z:z+80] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().to(device)    # .cuda()
        model.is_training=False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                    # 1.global模型 - RF Net
                    pred_part, _, _ = model(x_input, mask, None,None,None,None)
                    # 2.global模型 - E&D
                    # pred_part = model(x_input)
                    # 3.global模型 - HeMIS
                    # pred_part, _ = model(x_input.transpose(1,0).unsqueeze(2), mask, None,None,None,None)
                    # 4. global模型 - MMmodel
                    # pred_part, _, _, _ = model(x_input, mask, None, None, None, None)
                    pred[:, :, h:h+80, w:w+80, z:z+80] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print (msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
