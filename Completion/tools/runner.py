import torch
import torch.nn as nn
import os
from tools import builder
from utils import dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.Data_loader_ycb import YcbTest, YcbTrain
import open3d as o3d



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point




Test_data_dir = 'PATH_TO_THE_FILE/input/*/test'
Test_pcd_dir = 'PATH_TO_THE_FILE/gt'
Train_pcd_dir = 'PATH_TO_THE_FILE/gt'
Train_data_dir = 'PATH_TO_THE_FILE/input/*/train'


test_data = YcbTest(Test_data_dir, Test_pcd_dir, test_mode=True)
train_data = YcbTrain(Train_data_dir, Train_pcd_dir)

# NOTE THAT FOR THE REAL GRASPING EXP THE BATCH SIZE FOR THE TEST LOADER SHOULD BE 1, OTHERWISE IT CAN BE HIGHER
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=16)


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    _model = builder.model_builder(config.model)
    #wandb.watch(_model)

    _model.to(args.local_rank)
    start_epoch = 0
    best_loss = 300


    #PATH_TO_PRE_TRAINED_MODEL

   # builder.load_model(_model, 'PATH_TO_PRE_TRAINED_MODEL/MODEL.pth', logger=logger)

    _model = nn.DataParallel(_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(_model, config)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    _model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        _model.train()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])
        num_iter = 0
        _model.train()  # set model to training mode
        n_batches = len(train_loader)

        for idx, data in enumerate(tqdm(train_loader)):
            data_time.update(time.time() - batch_start_time)
            partial = data[0].float().cuda()
            gt = data[1].float().cuda()
            num_iter += 1
            output = _model(partial)
            coars_points = output[0]
            dense_points = output[1]
            sparse_loss = ChamferDisL1(coars_points, gt)
            dense_loss = ChamferDisL1(dense_points, gt)
            _loss = sparse_loss + dense_loss
            _loss.backward()

            # forward
            if num_iter == 1:
                num_iter = 0
                optimizer.step()
                _model.zero_grad()

            losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])

            n_itr = epoch * n_batches + idx
            if  train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        if train_writer is not None:


            metrics = validate(_model, test_loader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config,
                               logger=logger)

            # Save ckeckpoints best loss
            best_l = metrics[1]
            if best_l <= best_loss:
                print('the loss is:', best_l)
                best_loss = best_l
            if best_l <= best_loss:
                print('best model loss is saving')
                torch.save({
                    '_model': _model.module.state_dict() if args.distributed else _model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, os.path.join(args.experiment_path + 'model.pth'))

            print('lowest loss', best_loss)
            print('current loss', metrics[1])


    train_writer.close()
    val_writer.close()


def validate(_model, test_loader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=None):

    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    _model.eval()  # set model to eval mode


    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    with torch.no_grad():
        for idx, (data) in enumerate(tqdm(test_loader)):
            partial = data[0].float().cuda()
            gt = data[1].float().cuda()

            output = _model(partial)
            coarse_points = output[0]
            dense_points = output[1]


            sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
            dense_loss_l1 = ChamferDisL1(dense_points, gt)
            dense_loss_l2 = ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000,
                                dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt)

        for _, v in category_metrics.items():
            test_metrics.update(v.avg())

    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)

        #wandb.log({'Test/Loss/Epoch/Sparse': test_losses.avg(0), 'Epoch': epoch})
        #wandb.log({'Test/Loss/Epoch/Dense': test_losses.avg(2), 'Epoch': epoch})
        #wandb.log({'Test/Loss/Epoch/DenseL2': test_losses.avg(3), 'Epoch': epoch})
        print('DenseL2', test_losses.avg(3))

    return test_losses.avg(0), test_losses.avg(2), test_losses.avg(3)


def test_net(args, config):
    logger = get_logger(args.log_name)

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, ChamferDisL1, ChamferDisL1, ChamferDisL2, args, config, logger=logger)


def test(base_model, test_loader, ChamferDisL1, ChamferDisL2, args, config, logger=None):
    base_model.eval()

    with torch.no_grad():
        #PATH_TO_THE_PARTIAL_DATA. REMEMBER TO CHANGE THE PATH ALSO IN THE MAIN_GPD.PY FILE
        partial_ = o3d.io.read_point_cloud('PATH_TO_THE_PARTIAL_DATA/partial_pc.pcd')
        partial_ = np.asarray(partial_.points)
        partial = farthest_point_sample(partial_,2048)
        partial = partial[:,:3]


        centroid = np.mean(partial, axis=0)
        pc = partial - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        partial = pc.reshape(1, 2048, 3)

        partial = torch.tensor(partial)
        partial = partial.float().cuda()



        ret = base_model(partial)
        dense_points = ret[1]

        pcd = dense_points.cpu().squeeze()

        pcd_r = pcd * (m + (m / 6))
        pcd_r = pcd_r + centroid
        pcd_x = pcd * (m + (m/6))
        pcd_x = pcd_x + centroid


        pcdd = o3d.geometry.PointCloud()
        pcdd.points = o3d.utility.Vector3dVector(pcd_r.cpu().squeeze())

        #PATH_TO_THE_COMPLETED_DATA_REMEMBER TO CHANGE THE PATH ALSO IN THE MAIN_GPD.PY FILE
        o3d.io.write_point_cloud('PATH_TO_THE_COMPLETED_DATA/complete_pc.pcd',pcdd)
        np.savetxt('outputfile.xyz', pcd_r.cpu().squeeze())

        pcddx = o3d.geometry.PointCloud()
        pcddx.points = o3d.utility.Vector3dVector(pcd_x.cpu().squeeze())
        o3d.io.write_point_cloud('PATH_TO_THE_COMPLETED_DATA/complete_pc_x.pcd', pcddx)
        np.savetxt('outputfile.xyz', pcd_x.cpu().squeeze())

