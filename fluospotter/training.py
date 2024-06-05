"""Training functions."""
# pylint: disable=C0415

from typing import Dict
import datetime
import os, os.path
import platform
import torch
from tqdm import trange
from .datasets import Dataset
from .models import Model
import numpy as np
import random
import time
import json
import pdb
from monai.inferers import sliding_window_inference
from .data import get_loaders
from .optimizers import get_optimizer, get_scheduler
from .losses import get_loss
from .metrics import fast_bin_dice, fast_bin_auc
from skimage.filters import threshold_otsu
from .metrics import compute_segmentation_metrics
from .util import monitor_ram_usage

import threading

def init_tr_info():
    # I customize this function for each project.
    tr_info = dict()
    tr_info['tr_dscs'], tr_info['vl_dscs'] = [], []
    tr_info['tr_aucs'], tr_info['vl_aucs'] = [], []
    tr_info['tr_losses'], tr_info['vl_losses'] = [], []

    return tr_info


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, tr_loader, bs, acc_grad, loss_fn, optimizer, scheduler):
    model.train()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    n_opt_iters = 0
    with trange(len(tr_loader)) as t:
        step, n_elems, running_loss = 0, 0, 0
        for (i_batch, batch_data) in enumerate(tr_loader):  # load 1 scan from the training set
            n_samples = len(batch_data['seg'])  # nr of px x py x pz patches (see args.n_samples)
            for m in range(0, n_samples, bs):  # we loop over batch_data picking up bs patches at a time
                step += bs
                inputs, labels = (batch_data['img'][m:(m+bs)].to(device), batch_data['seg'][m:(m+bs)].to(device))
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss = loss / acc_grad
                loss.backward()
                if ((n_opt_iters + 1) % acc_grad == 0) or (n_opt_iters + 1 == len(tr_loader)):
                    # Update Optimizer
                    optimizer.step()
                    optimizer.zero_grad()
                n_opt_iters += 1
                lr = get_lr(optimizer)
                scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.detach().item() * inputs.shape[0]
                n_elems += inputs.shape[0]  # total nr of items processed
                run_loss = running_loss / n_elems

            t.set_postfix(LOSS_lr="{:.4f}/{:.6f}".format(run_loss, lr))
            t.update()


def validate(model, loader, loss_fn, slwin_bs=2):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    patch_size = model.patch_size
    # patch_size, slwin_bs = model.patch_size, 1
    dscs, aucs, losses = [], [], []
    with trange(len(loader)) as t:
        n_elems, running_dsc = 0, 0
        for val_data in loader:
            images, labels = val_data["img"].to(device), val_data["seg"]
            n_classes = labels.shape[1]
            preds = sliding_window_inference(images, patch_size, slwin_bs, model, overlap=0.1, mode='gaussian').cpu()
            del images
            loss = loss_fn(preds, labels)
            preds = preds.argmax(dim=1).squeeze().numpy()
            labels = labels.squeeze().numpy().astype(np.int8)

            dsc_score, auc_score = [], []
            #pdb.set_trace()
            for l in range(1,n_classes):
                dsc_score.append(fast_bin_dice(labels[l], preds == l))
                auc_score.append(fast_bin_auc(labels[l], preds == l, partial=True))
                if np.isnan(dsc_score[l]): dsc_score[l] = 0

            #preds = preds.sigmoid().squeeze().numpy()
            #th = threshold_otsu(preds)
            #preds = preds>th

            #labels = labels.squeeze().numpy().astype(bool)
            #dsc_score = fast_bin_dice(labels, preds)
            #auc_score = fast_bin_auc(labels, preds, partial=True)

            #if np.isnan(dsc_score): dsc_score = 0
            dscs.append(dsc_score)
            aucs.append(auc_score)
            losses.append(loss.item())
            n_elems += 1
            running_dsc += np.mean(dsc_score)
            run_dsc = running_dsc / n_elems
            t.set_postfix(DSC="{:.2f}".format(100 * run_dsc))
            t.update()

    return [100 * np.mean(np.array(dscs)), 100 * np.mean(np.array(aucs)), np.mean(np.array(losses))]


def evaluate(model, loader, cfg, slwin_bs=2):

    # Start the RAM usage monitoring thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_ram_usage, args=(stop_event,))
    monitor_thread.daemon = True
    monitor_thread.start()

    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    patch_size = tuple(map(int, cfg["patch_size"].split('/')))
    metrics = {}
    with trange(len(loader)) as t:
        for val_data in loader:
            images, labels = val_data["img"].to(device), val_data["seg"]
            preds = sliding_window_inference(images, patch_size, slwin_bs, model, overlap=0.1, mode='gaussian').cpu()
            del images
            preds = preds.argmax(dim=1).squeeze().numpy()
            labels = labels.squeeze().numpy().astype(np.int8)
            metrics = compute_segmentation_metrics(preds, labels, metrics)
            del preds
            del labels
            t.update()

    # Stop the monitoring thread
    stop_event.set()
    monitor_thread.join()
    return metrics

def set_tr_info(tr_info, epoch=0, ovft_metrics=None, vl_metrics=None, best_epoch=False):
    # I customize this for each project.
    # Here tr_info contains Dice Scores, AUCs, and loss values.
    # Also, and vl_metrics contain (in this order) dice, auc and loss
    if best_epoch:
        tr_info['best_tr_dsc'] = tr_info['tr_dscs'][-1]
        tr_info['best_vl_dsc'] = tr_info['vl_dscs'][-1]
        tr_info['best_tr_auc'] = tr_info['tr_aucs'][-1]
        tr_info['best_vl_auc'] = tr_info['vl_aucs'][-1]
        tr_info['best_tr_loss'] = tr_info['tr_losses'][-1]
        tr_info['best_vl_loss'] = tr_info['vl_losses'][-1]
        tr_info['best_epoch'] = epoch
    else:
        tr_info['tr_dscs'].append(ovft_metrics[0])
        tr_info['vl_dscs'].append(vl_metrics[0])
        tr_info['tr_aucs'].append(ovft_metrics[1])
        tr_info['vl_aucs'].append(vl_metrics[1])
        tr_info['tr_losses'].append(ovft_metrics[-1])
        tr_info['vl_losses'].append(vl_metrics[-1])

    return tr_info


def train_segmentation_model(model, optimizer, acc_grad, loss_fn, bs, tr_loader, ovft_loader, vl_loader, scheduler, metric, n_epochs, vl_interval, save_path, multiclass=False):
    best_metric, best_epoch = -1, 0
    tr_info = init_tr_info()
    for epoch in range(n_epochs):
        print('Epoch {:d}/{:d}'.format(epoch + 1, n_epochs))
        # train one cycle
        train_one_epoch(model, tr_loader, bs, acc_grad, loss_fn, optimizer, scheduler)
        if (epoch + 1) % vl_interval == 0:
            with torch.inference_mode():
                ovft_metrics = validate(model, ovft_loader, loss_fn)
                vl_metrics = validate(model, vl_loader, loss_fn)
            tr_info = set_tr_info(tr_info, epoch, ovft_metrics, vl_metrics)
            s = get_eval_string(tr_info, epoch)
            print(s)
            with open(os.path.join(save_path, 'train_log.txt'), 'a') as f: print(s, file=f)
            # check if performance was better than anyone before and checkpoint if so
            if metric =='DSC': curr_metric = tr_info['vl_dscs'][-1]
            elif metric == 'AUC': curr_metric = tr_info['vl_aucs'][-1]

            if curr_metric > best_metric:
                print('-------- Best {} attained. {:.2f} --> {:.2f} --------'.format(metric, best_metric, curr_metric))
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                best_metric, best_epoch = curr_metric, epoch + 1
                tr_info = set_tr_info(tr_info, epoch+1, best_epoch=True)
            else:
                print('-------- Best {} so far {:.2f} at epoch {:d} --------'.format(metric, best_metric, best_epoch))
    torch.save(model.state_dict(), os.path.join(save_path, 'last_model.pth'))
    del model, tr_loader, vl_loader
    torch.cuda.empty_cache()
    return tr_info

def train_model(
    model: Model,
    dataset: Dataset,
    model_type: str
) -> None:
    callbacks = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # Logging
    cfg = model.cfg
    if "save_path" in cfg:
        save_path = os.path.join(cfg["save_path"], 'experiments')
    else:
        save_path = os.path.join(dataset.data_dir, '../experiments')

    if model_type == "segmentation":
        save_path = os.path.join(save_path, 'segmentation')
        labels_path = dataset.segmentation_dir
    elif model_type == "punctaDetection":
        save_path = os.path.join(save_path, 'punctaDetection')
        labels_path = dataset.spots_dir
    else:
        raise ValueError("Model type {} not recognized".format(model_type))

    os.makedirs(save_path, exist_ok=True)

    #model_name = args.model_name
    #optimizer_choice = args.optimizer
    #lr, bs, ns, negs = args.lr, args.batch_size, args.n_samples, args.neg_samples
    #n_epochs, vl_interval, metric = args.n_epochs, args.vl_interval, args.metric
    #acc_grad, nw = args.acc_grad, args.num_workers

    #patch_size = args.patch_size.split('/')
    #patch_size = tuple(map(int, patch_size))

    print('* Instantiating a {} model for {}'.format(model.model_name, model_type))
    in_c = 1
    model = model.network
    model = model.to(device)
    print('* Creating Dataloaders, batch size = {}, samples/vol = {}, workers = {}'.format(cfg["batch_size"], cfg["n_samples"], cfg["num_workers"]))

    tr_loader, ovft_loader, vl_loader = get_loaders(data_path=dataset.data_dir, labels_path=labels_path, n_samples=int(cfg["n_samples"]), neg_samples=int(cfg["neg_samples"]),
                                                    patch_size=tuple(map(int, cfg["patch_size"].split('/'))), num_workers=int(cfg["num_workers"]), ovft_check=cfg["ovft_check"], depth_last=cfg["depth_last"],
                                                    n_classes=int(cfg["n_classes"]))

    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    opt_cfg = {}
    if "lr" in cfg:
        opt_cfg["learning_rate"] = float(cfg["lr"])
    if "weight_decay" in cfg:
        opt_cfg["weight_decay"] = float(cfg["weight_decay"])
    if "momentum" in cfg:
        opt_cfg["momentum"] = float(cfg["momentum"])

    optimizer = get_optimizer(cfg["optimizer"], opt_cfg, model.parameters())
    if cfg["cyclical_lr"]:
        scheduler_name = 'cosineAnnealingWarmRestarts'
        T = int(cfg["vl_interval"]) * len(tr_loader) * int(cfg["n_samples"]) // int(cfg["batch_size"])
    else:
        scheduler_name = 'cosineAnnealingLR'
        T = int(cfg["n_epochs"]) * len(tr_loader) * int(cfg["n_samples"]) // int(cfg["batch_size"])

    scheduler = get_scheduler(scheduler=scheduler_name, optimizer=optimizer, T=T, eta_min=0)
    loss_fn = get_loss(cfg["loss1"], cfg["loss2"], float(cfg["alpha1"]), float(cfg["alpha2"]))

    print('* Instantiating loss function {:.2f}*{} + {:.2f}*{}'.format(cfg["alpha1"], cfg["loss1"], cfg["alpha2"],
                                                                       cfg["loss2"]))
    print('* Starting to train\n', '-' * 10)
    start = time.time()

    if model_type == "segmentation":
        tr_info = train_segmentation_model(model, optimizer, cfg["acc_grad"], loss_fn, int(cfg["batch_size"]), tr_loader, ovft_loader, vl_loader, scheduler,
                          cfg["metric"], int(cfg["n_epochs"]), int(cfg["vl_interval"]), save_path, int(cfg["n_classes"]) > 1)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

    with (open(os.path.join(save_path, 'log.txt'), 'a') as f):
        print(
            'Best epoch = {}/{}: Tr/Vl DSC = {:.2f}/{:.2f} - Tr/Vl AUC = {:.2f}/{:.2f} - Tr/Vl LOSS = {:.4f}/{:.4f}\n'.format(
                tr_info['best_epoch'], cfg["n_epochs"],
                tr_info['best_tr_dsc'], tr_info['best_vl_dsc'], tr_info['best_tr_auc'], tr_info['best_vl_auc'],
                tr_info['best_tr_loss'], tr_info['best_vl_loss']), file=f)
        for j in range(cfg["n_epochs"] // cfg["vl_interval"]):
            s = get_eval_string(tr_info, epoch=j, finished=True, vl_interval=cfg["vl_interval"])
            print(s, file=f)
        print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

    print('Done. Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

    print('Finished.')


def get_eval_string(tr_info, epoch, finished=False, vl_interval=1):
    # I customize this function for each project.
    # Pretty prints first three values of train/val metrics to a string and returns it
    # Used also by the end of training (finished=True)
    ep_idx = len(tr_info['tr_dscs'])-1
    if finished:
        ep_idx = epoch
        epoch = (epoch+1) * vl_interval - 1

    s = 'Ep. {}: Train||Val DSC: {:5.2f}||{:5.2f} - AUC: {:5.2f}||{:5.2f} - Loss: {:.4f}||{:.4f}'.format(
        str(epoch+1).zfill(3), tr_info['tr_dscs'][ep_idx], tr_info['vl_dscs'][ep_idx],
              tr_info['tr_aucs'][ep_idx], tr_info['vl_aucs'][ep_idx],
              tr_info['tr_losses'][ep_idx], tr_info['vl_losses'][ep_idx])
    return s


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


'''
def run_experiment(cfg: Dict, pre_model: tf.keras.models.Model = None):
    """Run a training experiment.

    Configuration file can be generated using deepblink config.

    Args:
        cfg: Dictionary configuration file.
        pre_model: Pre-trained model if not training from scratch.
    """
    # Classes / functions
    dataset_class = get_from_module("deepblink.datasets", cfg["dataset"])
    model_class = get_from_module("deepblink.models", cfg["model"])
    network_fn = get_from_module("deepblink.networks", cfg["network"])
    optimizer_fn = get_from_module("deepblink.optimizers", cfg["optimizer"])
    loss_fn = get_from_module("deepblink.losses", cfg["loss"])

    # Arguments
    augmentation_args = cfg.get("augmentation_args", {})
    dataset_args = cfg.get("dataset_args", {})
    dataset = dataset_class(**dataset_args)
    network_args = (
        cfg.get("network_args", {}) if cfg.get("network_args", {}) is not None else {}
    )
    network_args["cell_size"] = dataset_args["cell_size"]
    train_args = cfg.get("train_args", {})

    model = model_class(
        augmentation_args=augmentation_args,
        dataset_args=dataset_args,
        dataset_cls=dataset,
        loss_fn=loss_fn,
        network_args=network_args,
        network_fn=network_fn,
        optimizer_fn=optimizer_fn,
        train_args=train_args,
        pre_model=pre_model,
    )

    cfg["system"] = {
        "gpus": tf.config.list_logical_devices("GPU"),
        "version": platform.version(),
        "platform": platform.platform(),
    }

    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"{now}_{cfg['run_name']}"

    use_wandb = cfg["use_wandb"]
    if use_wandb:
        try:
            import wandb

            if wandb.__version__ <= "0.10.03":
                raise AssertionError
        except (ModuleNotFoundError, AttributeError, AssertionError):
            raise ImportError(
                (
                    "To support conda packages we don't ship deepBlink with wandb. "
                    "Please install any using pip: 'pip install \"wandb>=0.10.3\"'"
                )
            )

        # pylint:disable=E1101
        wandb.init(name=run_name, project=cfg["name"], config=cfg)

    model = train_model(model, dataset, cfg, run_name, use_wandb)

    if use_wandb:
        wandb.join()  # pylint:disable=E1101'''
