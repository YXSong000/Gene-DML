import os
import random
import argparse
from datetime import datetime
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from pathlib import Path
from torch.optim.adam import Adam

from dml import GeneDML
from st_data import STDataset
from utils import collate_fn, load_config
import torchmetrics
from torchmetrics.regression.pearson import PearsonCorrCoef
from torchmetrics.regression.concordance import ConcordanceCorrCoef
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.mae import MeanAbsoluteError
from torchmetrics.regression.explained_variance import ExplainedVariance
import matplotlib.pyplot as plt

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='her2st/dml', help='logger path.')
    parser.add_argument('--gpu', type=int, default=3, help='gpu id')
    parser.add_argument('--mode', type=str, default='train', help='train / test / external_test')
    parser.add_argument('--test_name', type=str, default='NCBI463', help='dataset name:{"10x_breast_ff1","10x_breast_ff2", "10x_breast_ff3", "NCBI463", "NCBI464"}.')
    parser.add_argument('--exp_id', type=int, default=0, help='')
    parser.add_argument('--fold', type=int, default=0, help='')
    parser.add_argument('--use_kmeans', action="store_true")
    parser.add_argument('--k', type=int, default=18, help='number of clusters')
    parser.add_argument('--model_path', type=str, default='path/to/log', help='path to save models')
    parser.add_argument('--model_size', type=str, default='base', help='base / large')
    parser.add_argument('--lambda_cld', type=float, default=1.0, help='lambda for CLD loss')
    parser.add_argument('--cld_t', type=float, default=0.1, help='temperature for CLD loss')

    args = parser.parse_args()
    
    return args

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(model, train_loader, optimizer, scheduler, scaler, epoch, alpha, device):
    model.train()
    batch_losses = []
    for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        patch, exp, pid, sid, wsi, position, neighbor, mask = batch

        # Move only tensors to the GPU
        patch = patch.to(device) if isinstance(patch, torch.Tensor) else patch
        exp = exp.to(device) if isinstance(exp, torch.Tensor) else exp
        pid = pid.to(device) if isinstance(pid, torch.Tensor) else pid
        sid = sid.to(device) if isinstance(sid, torch.Tensor) else sid
        neighbor = neighbor.to(device) if isinstance(neighbor, torch.Tensor) else neighbor
        mask = mask.to(device) if isinstance(mask, torch.Tensor) else mask
        
        # wsi and position are assumed to be lists or other non-tensor types
        # If they are tensors, move them to the GPU
        wsi = [item.to(device) if isinstance(item, torch.Tensor) else item for item in wsi]
        position = [item.to(device) if isinstance(item, torch.Tensor) else item for item in position]
        optimizer.zero_grad()

        with autocast(device_type=device):

            loss, loss_dict = model(patch, exp, wsi, position, neighbor, mask, pid, sid, test=False, use_kmeans=cfg.use_kmeans, k=cfg.k)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        loss = float(loss)
        batch_losses.append(loss)

        print(f"Epoch {epoch}, Batch {i}, Loss: {loss}")
        wandb.log({'training_loss': loss})
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr']})
        wandb.log({'contrastive_loss': loss_dict['contrastive_loss']})
        wandb.log({'prediction_loss': loss_dict['prediction_loss']})
        wandb.log({'target_loss': loss_dict['target_pred_loss']})
        wandb.log({'neighbor_loss': loss_dict['neighbor_pred_loss']})
        wandb.log({'global_loss': loss_dict['global_pred_loss']})
    
    scheduler.step()
    print(f"Finished Epoch {epoch}, Average Loss: {np.mean(batch_losses)}")
    wandb.log({'epoch_average_training_loss': np.mean(batch_losses)})

def get_meta(name):
    if '10x_breast' in name[0]:
        patient = name[0]
        data = "test"
    else:
        name = name[0]
        data = name.split("+")[1]
        patient = name.split("+")[0]
        
        if data == 'her2st':
            patient = patient[0]
        elif data == 'stnet':
            data = "stnet"
            patient = patient.split('_')[0]
            if patient in ['BC23277', 'BC23287', 'BC23508']:
                patient = 'BC1'
            elif patient in ['BC24105', 'BC24220', 'BC24223']:
                patient = 'BC2'
            elif patient in ['BC23803', 'BC23377', 'BC23895']:
                patient = 'BC3'
            elif patient in ['BC23272', 'BC23288', 'BC23903']:
                patient = 'BC4'
            elif patient in ['BC23270', 'BC23268', 'BC23567']:
                patient = 'BC5'
            elif patient in ['BC23269', 'BC23810', 'BC23901']:
                patient = 'BC6'
            elif patient in ['BC23209', 'BC23450', 'BC23506']:
                patient = 'BC7'
            elif patient in ['BC23944', 'BC24044']:
                patient = 'BC8'
        elif data == 'skin':
            patient = patient.split('_')[0]
    return patient

def plot_hexbin(x, y, values, title, save_name):
    min_val = values.min()
    max_val = values.max()
    normalized_values = (values - min_val) / (max_val - min_val)
    plt.figure()
    hb = plt.hexbin(x, y, C=normalized_values, gridsize=18, cmap="coolwarm", edgecolors='black', mincnt=1, linewidths=0.2)
    plt.axis("off")  # Remove axes
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(hb)
    cbar.set_label("Gene Expression Level (Normalized)")
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0)


def validation(cfg, model, test_loader, epoch, device, test=True, use_kmeans=True, k=10):
    model.eval()
    batch_outputs = []
    pcc_all_hpg = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Validation Epoch {epoch}")):
            patch, exp, _, wsi, position, name, neighbor, mask, centers = batch

            # Move tensors to the GPU
            patch = patch.squeeze().to(device) if isinstance(patch, torch.Tensor) else patch
            exp = exp.squeeze().to(device) if isinstance(exp, torch.Tensor) else exp
            neighbor = neighbor.squeeze().to(device) if isinstance(neighbor, torch.Tensor) else neighbor
            mask = mask.squeeze().to(device) if isinstance(mask, torch.Tensor) else mask
            wsi = wsi.to(device) if isinstance(wsi, torch.Tensor) else wsi
            position = position.to(device) if isinstance(position, torch.Tensor) else position

            patient = get_meta(name)

            loss, loss_dict = model(patch, exp, wsi, position, neighbor, mask)
            target_pred = loss_dict['predictions']['target']
            neighbor_pred = loss_dict['predictions']['neighbor']
            global_pred = loss_dict['predictions']['global']
            pred = loss_dict['predictions']['fusion']
            target_pred = target_pred.view_as(exp)
            neighbor_pred = neighbor_pred.view_as(exp)
            global_pred = global_pred.view_as(exp)
            pred = pred.view_as(exp)
            mse = F.mse_loss(pred, exp)
            target_mse = F.mse_loss(target_pred, exp)
            neighbor_mse = F.mse_loss(neighbor_pred, exp)
            global_mse = F.mse_loss(global_pred, exp)
            mae = F.l1_loss(pred, exp)
            target_mae = F.l1_loss(target_pred, exp)
            neighbor_mae = F.l1_loss(neighbor_pred, exp)
            global_mae = F.l1_loss(global_pred, exp)
            pred = pred.cpu()
            target_pred = target_pred.cpu()
            neighbor_pred = neighbor_pred.cpu()
            global_pred = global_pred.cpu()
            exp = exp.cpu()
            
            metrics = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = 250),
                                                            ConcordanceCorrCoef(num_outputs = 250),
                                                            MeanSquaredError(num_outputs = 250),
                                                            MeanAbsoluteError(num_outputs = 250),
                                                            ExplainedVariance(),
                                                            ])
            metrics_50 = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = 50),
                                                            ConcordanceCorrCoef(num_outputs = 50),
                                                            MeanSquaredError(num_outputs = 50),
                                                            MeanAbsoluteError(num_outputs = 50),
                                                            ExplainedVariance(),
                                                            ])
            test_metrics = metrics.clone(prefix = 'test_')
            test_metrics_50 = metrics_50.clone(prefix = 'test_')

            test_result = test_metrics(pred, exp)
            rr = test_result['test_PearsonCorrCoef'].mean()
            mse = test_result['test_MeanSquaredError']
            mae = test_result['test_MeanAbsoluteError']

            target_test_result = test_metrics(target_pred, exp)
            target_rr = target_test_result['test_PearsonCorrCoef'].mean()
            target_mse = target_test_result['test_MeanSquaredError']
            target_mae = target_test_result['test_MeanAbsoluteError']

            neighbor_test_result = test_metrics(neighbor_pred, exp)
            neighbor_rr = neighbor_test_result['test_PearsonCorrCoef'].mean()
            neighbor_mse = neighbor_test_result['test_MeanSquaredError']
            neighbor_mae = neighbor_test_result['test_MeanAbsoluteError']

            global_test_result = test_metrics(global_pred, exp)
            global_rr = global_test_result['test_PearsonCorrCoef'].mean()
            global_mse = global_test_result['test_MeanSquaredError']
            global_mae = global_test_result['test_MeanAbsoluteError']


            pcc_rank = torch.argsort(torch.argsort(test_result['test_PearsonCorrCoef'], dim=-1), dim=-1) + 1
            np.save(f"results/CFC/her2st/pcc_rank{cfg.fold}.npy", pcc_rank.numpy())



        

            if os.path.exists("results/CFC/her2st/idx_top.npy"):
                idx_top = np.load("results/CFC/her2st/idx_top.npy")
                test_metrics_50 = metrics_50.clone(prefix = 'test_')
                test_metric_hpg = test_metrics_50(pred[:, idx_top], exp[:, idx_top])
                pcc_all_hpg.append(test_metric_hpg['test_PearsonCorrCoef'].mean())

            batch_outputs.append({"mse":mse, "mae":mae, "target_mse":target_mse, "target_mae":target_mae, "target_rr":target_rr, "rr":rr, "neighbor_mse":neighbor_mse, "neighbor_mae":neighbor_mae, "neighbor_rr":neighbor_rr, "global_mse":global_mse, "global_mae":global_mae, "global_rr":global_rr, "loss":loss})

        avg_pcc_rank = []
        for pcc_file in os.listdir(f"results/CFC/her2st"):
            if "pcc_rank" in pcc_file:
                print("results/CFC/her2st/"+pcc_file)
                pcc_rank = np.load("results/CFC/her2st/"+pcc_file)
                avg_pcc_rank.append(pcc_rank)
        
        avg_pcc_rank = np.mean(avg_pcc_rank, axis=0)
        idx_top = np.argsort(avg_pcc_rank)[::-1][:50]
        print(idx_top)

        np.save(f"results/CFC/her2st/idx_top.npy", idx_top)

        mse = torch.stack(
            [x["mse"] for x in batch_outputs]).mean()
        mae = torch.stack(
            [x["mae"] for x in batch_outputs]).mean()
        rr = torch.stack(
            [x["rr"] for x in batch_outputs]).mean()
        target_mse = torch.stack(
            [x["target_mse"] for x in batch_outputs]).mean()
        target_rr = torch.stack(
            [x["target_rr"] for x in batch_outputs]).mean()
        target_mae = torch.stack(
            [x["target_mae"] for x in batch_outputs]).mean()
        neighbor_mse = torch.stack(
            [x["neighbor_mse"] for x in batch_outputs]).mean()
        neighbor_mae = torch.stack(
            [x["neighbor_mae"] for x in batch_outputs]).mean()
        neighbor_rr = torch.stack(
            [x["neighbor_rr"] for x in batch_outputs]).mean()
        global_mse = torch.stack(
            [x["global_mse"] for x in batch_outputs]).mean()
        global_rr = torch.stack(
            [x["global_rr"] for x in batch_outputs]).mean()
        global_mae = torch.stack(
            [x["global_mae"] for x in batch_outputs]).mean()
        loss = torch.stack(
            [x["loss"] for x in batch_outputs]).mean()
        
        print(f'Epoch {epoch} valid_loss: {mse}, patient: {patient}')
        wandb.log({'val_mse': mse})
        wandb.log({'val_mae': mae})
        wandb.log({'val_rr': rr})
        wandb.log({'val_target_mse': target_mse})
        wandb.log({'val_neighbor_mse': neighbor_mse})
        wandb.log({'val_global_mse': global_mse})
        wandb.log({'val_target_mae': target_mae})
        wandb.log({'val_neighbor_mae': neighbor_mae})
        wandb.log({'val_global_mae': global_mae})
        wandb.log({'val_target_rr': target_rr})
        wandb.log({'val_neighbor_rr': neighbor_rr})
        wandb.log({'val_global_rr': global_rr})
        wandb.log({'val_loss': loss})
        wandb.log({'val_rr_50': np.mean(pcc_all_hpg)})
        return rr, patient

def test(cfg, model, ckpt, test_loader, device, test=True, use_kmeans=True, k=10):
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    batch_outputs = []
    pcc_all_hpg = []
    all_centers = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            patch, exp, _, wsi, position, name, neighbor, mask, centers = batch

            # Move tensors to the GPU
            patch = patch.squeeze().to(device) if isinstance(patch, torch.Tensor) else patch
            exp = exp.squeeze().to(device) if isinstance(exp, torch.Tensor) else exp
            neighbor = neighbor.squeeze().to(device) if isinstance(neighbor, torch.Tensor) else neighbor
            mask = mask.squeeze().to(device) if isinstance(mask, torch.Tensor) else mask
            wsi = wsi.to(device) if isinstance(wsi, torch.Tensor) else wsi
            position = position.to(device) if isinstance(position, torch.Tensor) else position
            centers=centers.squeeze()
            query_centers = torch.tensor(centers)
            print("query_centers shape: ", query_centers.shape)
            

            patient = get_meta(name)

            loss_dict = model(patch, exp, wsi, position, neighbor, mask, test=True)
            target_pred = loss_dict['predictions']['target']
            neighbor_pred = loss_dict['predictions']['neighbor']
            global_pred = loss_dict['predictions']['global']
            pred = loss_dict['predictions']['fusion']
            target_pred = target_pred.view_as(exp)
            neighbor_pred = neighbor_pred.view_as(exp)
            global_pred = global_pred.view_as(exp)
            pred = pred.view_as(exp)
            mse = F.mse_loss(pred, exp)
            target_mse = F.mse_loss(target_pred, exp)
            neighbor_mse = F.mse_loss(neighbor_pred, exp)
            global_mse = F.mse_loss(global_pred, exp)
            mae = F.l1_loss(pred, exp)
            target_mae = F.l1_loss(target_pred, exp)
            neighbor_mae = F.l1_loss(neighbor_pred, exp)
            global_mae = F.l1_loss(global_pred, exp)
            pred = pred.cpu()
            target_pred = target_pred.cpu()
            neighbor_pred = neighbor_pred.cpu()
            global_pred = global_pred.cpu()
            exp = exp.cpu()
            
            metrics = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = 250),
                                                            ConcordanceCorrCoef(num_outputs = 250),
                                                            MeanSquaredError(num_outputs = 250),
                                                            MeanAbsoluteError(num_outputs = 250),
                                                            ExplainedVariance(),
                                                            ])
            metrics_50 = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = 50),
                                                            ConcordanceCorrCoef(num_outputs = 50),
                                                            MeanSquaredError(num_outputs = 50),
                                                            MeanAbsoluteError(num_outputs = 50),
                                                            ExplainedVariance(),
                                                            ])
            test_metrics = metrics.clone(prefix = 'test_')
            test_metrics_50 = metrics_50.clone(prefix = 'test_')

            test_result = test_metrics(pred, exp)
            rr = test_result['test_PearsonCorrCoef']
            mse = test_result['test_MeanSquaredError']
            mae = test_result['test_MeanAbsoluteError']

            target_test_result = test_metrics(target_pred, exp)
            target_rr = target_test_result['test_PearsonCorrCoef'].mean()
            target_mse = target_test_result['test_MeanSquaredError']
            target_mae = target_test_result['test_MeanAbsoluteError']

            neighbor_test_result = test_metrics(neighbor_pred, exp)
            neighbor_rr = neighbor_test_result['test_PearsonCorrCoef'].mean()
            neighbor_mse = neighbor_test_result['test_MeanSquaredError']
            neighbor_mae = neighbor_test_result['test_MeanAbsoluteError']

            global_test_result = test_metrics(global_pred, exp)
            global_rr = global_test_result['test_PearsonCorrCoef'].mean()
            global_mse = global_test_result['test_MeanSquaredError']
            global_mae = global_test_result['test_MeanAbsoluteError']

            pcc_rank = torch.argsort(torch.argsort(test_result['test_PearsonCorrCoef'], dim=-1), dim=-1) + 1
            np.save(f"results/CFC/her2st/pcc_rank{cfg.fold}.npy", pcc_rank.numpy())
            print(pcc_rank)
            pcc_rank = np.load(f"results/CFC/her2st/pcc_rank{cfg.fold}.npy")
            idx_top = np.argsort(pcc_rank)[::-1][:50]
            print(idx_top)

            x_pixel=query_centers[:, 0].cpu().tolist()
            y_pixel=query_centers[:, 1].cpu().tolist()
            assert len(x_pixel) == len(y_pixel) == pred.shape[0]
            print('x_pixel: ', x_pixel)
            print('y_pixel: ', y_pixel)
            # visualize top gene count prediction
            for gene in [64, 41]:
                exp_value = pred[:, gene]
                true_value = exp[:, gene]
                r_value = rr[gene]
                plot_hexbin(x_pixel, y_pixel, exp_value, f'Gene {gene}', f'{cfg.model_path}/{i}_gene_{gene}_{patient}_{r_value}.svg')
                plot_hexbin(x_pixel, y_pixel, true_value, f'Gene {gene}', f'{cfg.model_path}/{i}_gene_{gene}_true_{patient}_{r_value}.svg')



            


        

            if os.path.exists("results/CFC/her2st/idx_top.npy"):
                idx_top = np.load("results/CFC/her2st/idx_top.npy")
                test_metrics_50 = metrics_50.clone(prefix = 'test_')
                test_metric_hpg = test_metrics_50(pred[:, idx_top], exp[:, idx_top])
                pcc_all_hpg.append(test_metric_hpg['test_PearsonCorrCoef'].mean())

            batch_outputs.append({"mse":mse, "mae":mae, "target_mse":target_mse, "target_mae":target_mae, "target_rr":target_rr, "rr":rr, "neighbor_mse":neighbor_mse, "neighbor_mae":neighbor_mae, "neighbor_rr":neighbor_rr, "global_mse":global_mse, "global_mae":global_mae, "global_rr":global_rr})

        avg_pcc_rank = []
        for pcc_file in os.listdir(f"results/CFC/her2st"):
            if "pcc_rank" in pcc_file:
                print("results/CFC/her2st/"+pcc_file)
                pcc_rank = np.load("results/CFC/her2st/"+pcc_file)
                avg_pcc_rank.append(pcc_rank)
        
        avg_pcc_rank = np.mean(avg_pcc_rank, axis=0)
        idx_top = np.argsort(avg_pcc_rank)[::-1][:50]
        print(idx_top)

        np.save(f"results/CFC/her2st/idx_top.npy", idx_top)

        mse = torch.stack(
            [x["mse"] for x in batch_outputs]).mean()
        mae = torch.stack(
            [x["mae"] for x in batch_outputs]).mean()
        rr = torch.stack(
            [x["rr"] for x in batch_outputs]).mean()
        target_mse = torch.stack(
            [x["target_mse"] for x in batch_outputs]).mean()
        target_rr = torch.stack(
            [x["target_rr"] for x in batch_outputs]).mean()
        target_mae = torch.stack(
            [x["target_mae"] for x in batch_outputs]).mean()
        neighbor_mse = torch.stack(
            [x["neighbor_mse"] for x in batch_outputs]).mean()
        neighbor_mae = torch.stack(
            [x["neighbor_mae"] for x in batch_outputs]).mean()
        neighbor_rr = torch.stack(
            [x["neighbor_rr"] for x in batch_outputs]).mean()
        global_mse = torch.stack(
            [x["global_mse"] for x in batch_outputs]).mean()
        global_rr = torch.stack(
            [x["global_rr"] for x in batch_outputs]).mean()
        global_mae = torch.stack(
            [x["global_mae"] for x in batch_outputs]).mean()
        return rr, patient
    

def test_external(cfg, model, checkpoint_path, test_loader, device):
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        batch_outputs = []
        pcc_all_hpg = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                patch, exp, sid, wsi, position, name, neighbor, mask = batch

                # Move tensors to the GPU
                patch = patch.squeeze().to(device) if isinstance(patch, torch.Tensor) else patch
                exp = exp.squeeze().to(device) if isinstance(exp, torch.Tensor) else exp
                sid = sid.squeeze().to(device) if isinstance(sid, torch.Tensor) else sid
                wsi = wsi.to(device) if isinstance(wsi, torch.Tensor) else wsi
                position = position.to(device) if isinstance(position, torch.Tensor) else position
                neighbor = neighbor.squeeze().to(device) if isinstance(neighbor, torch.Tensor) else neighbor
                mask = mask.squeeze().to(device) if isinstance(mask, torch.Tensor) else mask

                wsi = wsi[0].unsqueeze(0)
                position = position[0]
                
                patches = patch.split(512, dim=0)
                neighbors = neighbor.split(512, dim=0)
                masks = mask.split(512, dim=0)
                sids = sid.split(512, dim=0)
                
                preds  = []
                for patch, neighbor, mask, sid in zip(patches, neighbors, masks, sids):
                    loss_dict = model(patch, exp, wsi, position, neighbor, mask, sid=sid, test=True)
                    target_pred = loss_dict['predictions']['target']
                    neighbor_pred = loss_dict['predictions']['neighbor']
                    global_pred = loss_dict['predictions']['global']
                    pred = loss_dict['predictions']['fusion']
                    pred = pred.cpu()
                    target_pred = target_pred.cpu()
                    neighbor_pred = neighbor_pred.cpu()
                    global_pred = global_pred.cpu()
                    exp = exp.cpu()

                    
                    preds.append(pred)
                    
                preds = torch.cat(preds, dim=0)
                
                
                ind_match = np.load(f'{cfg.DATASET.data_dir}/test/{name[0]}/ind_match.npy', allow_pickle=True)
                model.num_genes = len(ind_match)
                pred = preds[:,ind_match]
            pred = pred.cpu().numpy()
            exp = exp.cpu().numpy()
            metrics = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = model.num_genes),
                                                                ConcordanceCorrCoef(num_outputs = model.num_genes),
                                                                MeanSquaredError(num_outputs = model.num_genes),
                                                                MeanAbsoluteError(num_outputs = model.num_genes),
                                                                ExplainedVariance(),
                                                                ])
            metrics_50 = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = 50),
                                                            ConcordanceCorrCoef(num_outputs = 50),
                                                            MeanSquaredError(num_outputs = 50),
                                                            MeanAbsoluteError(num_outputs = 50),
                                                            ExplainedVariance(),
                                                            ])
            test_metrics = metrics.clone(prefix = 'test_')
            test_metrics_50 = metrics_50.clone(prefix = 'test_')
            pred = torch.Tensor(pred)
            exp = torch.Tensor(exp)

            test_result = test_metrics(pred, exp)
            rr = test_result['test_PearsonCorrCoef'].mean()
            mse = test_result['test_MeanSquaredError']
            mae = test_result['test_MeanAbsoluteError']

            pcc_rank = torch.argsort(torch.argsort(test_result['test_PearsonCorrCoef'], dim=-1), dim=-1) + 1
            pcc_rank = pcc_rank.numpy()
            idx_top = np.argsort(pcc_rank)[::-1][:50]
            np.save('idx_top.npy', idx_top)
            idx_top = np.load('idx_top.npy')
            print(idx_top)
            test_metrics_50 = metrics_50.clone(prefix = 'test_')
            # Fix: Create copies to avoid negative strides
            pred_top = pred[:, idx_top]
            exp_top = exp[:, idx_top]
            test_metric_hpg = test_metrics_50(pred_top, exp_top)
            pcc_all_hpg.append(test_metric_hpg['test_PearsonCorrCoef'].mean())

            
            patient = get_meta(name)

            batch_outputs.append({"MSE": mse, "MAE": mae, "corr": rr})

        avg_mse = torch.stack(
            [x["MSE"] for x in batch_outputs]).nanmean()

        avg_mae = torch.stack(
            [x["MAE"] for x in batch_outputs]).nanmean()

        avg_corr = torch.stack(
            [x["corr"] for x in batch_outputs]).nanmean(0)


        print(f'checkpoint: {checkpoint_path}')
        print(f'MSE: {avg_mse}, MAE: {avg_mae}, R: {avg_corr.nanmean()}, patient: {patient}, PCC_50: {np.mean(pcc_all_hpg)}')
        return avg_mse, avg_mae, avg_corr.nanmean(), patient
    except:
        print(f'Error loading checkpoint: {checkpoint_path}')



def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def list_ckpt_files(root_dir, data_type, fold=0):
    ckpt_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ckpt') and data_type in filename and fold == int(filename.split('-')[0]):
                ckpt_files.append(os.path.join(dirpath, filename))
    return ckpt_files 

def main(cfg, fold=0):    
    
    seed=cfg.GENERAL.seed
    name=cfg.MODEL.name
    data=cfg.DATASET.type
    batch_size=cfg.TRAINING.batch_size
    num_epochs=cfg.TRAINING.num_epochs
    mode = cfg.GENERAL.mode
    gpu = f"cuda:{cfg.GENERAL.gpu}" if torch.cuda.is_available() and torch.cuda.device_count() > int(cfg.GENERAL.gpu) else "cpu"
    exp_id = cfg.GENERAL.exp_id

        

    # print time
    print('Start training/testing: {}'.format(datetime.now()))

    # Train or test model
    if mode == 'train':
        
        # Load model
        model = GeneDML(
            lambda_cld=cfg.lambda_cld,
            cld_t=cfg.cld_t,
            contrastive_weight=1.0,
            prediction_weight=1.0,
            training_mode='both',  # Use 'both' for joint training, 'prediction' for supervised only
            size=cfg.model_size,
            data_name=cfg.DATASET.type
        ).to(gpu)
        
        best_loss = np.inf
        best_cor = -1
        epochs_no_improve = 0
        patience = cfg.TRAINING.early_stopping.patience

        trainset = STDataset(mode='train', fold=fold, **cfg.DATASET)
        train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1, pin_memory=True, shuffle=True)
        testset = STDataset(mode='test', fold=fold, **cfg.DATASET)
        test_loader = DataLoader(testset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

        log_name=f'{fold}-{name}-{data}-{seed}-{exp_id}-lambda_cld={cfg.lambda_cld}-temperature={cfg.temperature}'
        os.makedirs(cfg.GENERAL.log_path, exist_ok=True)
        cfg.GENERAL.model_path = os.path.join(cfg.GENERAL.log_path, cfg.GENERAL.current_day, log_name)
        if not os.path.exists(cfg.GENERAL.model_path):
            os.makedirs(cfg.GENERAL.model_path)

        optimizer = Adam(model.parameters(), lr=cfg.MODEL.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)
        scaler = GradScaler()

        for epoch in range(num_epochs):
            train(model, train_loader, optimizer, scheduler, scaler, epoch, alpha=0.3, device=gpu,) # model change to online_model & target_model
            rr, patient = validation(cfg, model, test_loader, epoch, device=gpu)

            os.makedirs(f"results/{cfg.MODEL.name}/{cfg.DATASET.type}", exist_ok=True)
            if rr > best_cor:
                print('Saving model: improved PCC from {} to {}'.format(best_cor, rr))
                best_cor = rr

                model_fname = cfg.GENERAL.model_path + '/' f'{log_name}-epoch={epoch:03d}-valid_pcc={rr:.4f}.ckpt'
                for file in os.listdir(cfg.GENERAL.model_path):
                    os.remove(os.path.join(cfg.GENERAL.model_path, file))
                torch.save(model.state_dict(), model_fname)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"No improvement in {patience} epochs. Early stopping at epoch {epoch}")
                    break

        del trainset, train_loader
        del testset, test_loader
        del model
        gc.collect()
        torch.cuda.empty_cache()

    elif mode == 'external_test':
        model = GeneDML(size=cfg.model_size, data_name=cfg.DATASET.type, test=True).to(gpu)
        ckpt_files = list_ckpt_files(cfg.GENERAL.model_path, cfg.DATASET.type)
        testset = STDataset(mode='external_test', fold=fold, test_data=cfg.test_name, data_dir='/srv2/yson2999/data', type=cfg.DATASET.type, num_neighbors=5, t_global_dir='gt_features_uni_small_224', n_global_dir='gn_features_uni_small', neighbor_dir='n_features_uni_small', radius=224, use_pyvips=False)
        test_loader = DataLoader(testset, batch_size=1, num_workers=1, pin_memory=False, shuffle=False)
        print(f'number of ckpt files: {len(ckpt_files)}')
        print(f'ckpt_files: {ckpt_files}')
        for ckpt in ckpt_files:
            test_external(cfg, model, ckpt, test_loader, device=gpu)
    elif mode == 'test':
        model = GeneDML(size=cfg.model_size, data_name=cfg.DATASET.type, test=True).to(gpu)
        ckpt_files = list_ckpt_files(cfg.GENERAL.model_path, cfg.DATASET.type, fold=fold)
        testset = STDataset(mode='test', fold=fold, **cfg.DATASET)
        test_loader = DataLoader(testset, batch_size=1, num_workers=1, shuffle=False)
        for ckpt in ckpt_files:
            print(f"Testing model: {ckpt}")
            test(cfg, model, ckpt, test_loader, device=gpu)


        
        
    else:
        raise Exception("Invalid mode")
    
    # return model

if __name__ == '__main__':
    
    print('Start whole process: {}'.format(datetime.now()))

    args = get_parse()   
    cfg = load_config(args.config_name)

    seed = cfg.GENERAL.seed
    fix_seed(seed)
    
    cfg.GENERAL.test_name = args.test_name
    cfg.GENERAL.exp_id = args.exp_id
    cfg.GENERAL.gpu = args.gpu
    cfg.GENERAL.model_path = args.model_path
    cfg.GENERAL.mode = args.mode
    cfg.use_kmeans = args.use_kmeans
    cfg.k = args.k
    cfg.model_size = args.model_size
    cfg.lambda_cld = args.lambda_cld
    cfg.cld_t = args.cld_t
    
    current_day = datetime.now().strftime('%Y-%m-%d')
    cfg.GENERAL.current_day = current_day
    print(f'number of gpus: {torch.cuda.device_count()}')
    if args.mode == 'train':
        
        num_k = cfg.TRAINING.num_k     
        for fold in range(num_k):
            cfg.fold = fold
            wandb.init(job_type='training',
                    mode="online",
                    config=cfg,
                    project="Gene-DML",
                    entity="your_entity", # your wandb entity
                    group=args.config_name.split('/')[0]+'_'+args.mode,
                    name=args.config_name.split('/')[0]+'_'+args.mode+str(fold),
                    tags=[args.config_name.split('/')[0]]) 
            main(cfg, fold=fold)
            wandb.finish()
    else:
        cfg.test_name = args.test_name
        main(cfg, args.fold)

    print("Process Finished")