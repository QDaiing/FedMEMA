import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--datapath', default='/disk3/qd/edFL/data/BRATS2018_Training_none_npy/', type=str)
    parser.add_argument('--dataname', default='BRATS2020', type=str)
    parser.add_argument('--chose_modal', default='all', type=str)
    parser.add_argument('--num_class', default=4, type=int)
    parser.add_argument('--save_root', default='results', type=str)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    # parser.add_argument('--iter_per_epoch', default=150, type=int)
    # parser.add_argument('--region_fusion_start_epoch', default=20, type=int)
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--visualize', default=True)
    parser.add_argument('--deterministic', default=True)
    parser.add_argument('--seed', default=42, type=int)

    # FL Settings
    parser.add_argument('--gpus', default='7,6,5,4,3,2,1', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--c_rounds', type=int, default=300, help="number of rounds of training and communication")
    parser.add_argument('--start_round', type=int, default=0, help="number of rounds of training and communication")

    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--global_ep', type=int, default=1, help="the number of global epochs: E")
    parser.add_argument('--client_num', type=int, default=4, help="number of users: K")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    # files
    parser.add_argument('--train_file', type=dict, 
                default={ # 0:"/disk3/qd/edFL/data/BRATS2018_Training_none_npy/split_4c_1g_1v_1t_dup/glb.csv", 
                'glb':"/disk3/qd/edFL/data/BRATS2018_Training_none_npy/split_4c_1g_1v_1t/glb.csv", 
                1:"/disk3/qd/edFL/data/BRATS2018_Training_none_npy/split_4c_1g_1v_1t_dup/c1.csv", 
                2:"/disk3/qd/edFL/data/BRATS2018_Training_none_npy/split_4c_1g_1v_1t_dup/c2.csv", 
                3:"/disk3/qd/edFL/data/BRATS2018_Training_none_npy/split_4c_1g_1v_1t_dup/c3.csv", 
                4:"/disk3/qd/edFL/data/BRATS2018_Training_none_npy/split_4c_1g_1v_1t_dup/c4.csv"})
    parser.add_argument('--valid_file', type=str, default="/disk3/qd/edFL/data/BRATS2018_Training_none_npy/split_4c_1g_1v_1t_dup/val.csv")
    parser.add_argument('--test_file', type=str, default="/disk3/qd/edFL/data/BRATS2018_Training_none_npy/split_4c_1g_1v_1t_dup/test.csv")

    # 说明
    parser.add_argument('--version', type=str, default='debug', help='to explain the experiment set up')

    args = parser.parse_args()
    return args