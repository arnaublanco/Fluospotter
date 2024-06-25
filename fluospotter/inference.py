"""Model prediction / inference functions."""
import pdb
import numpy as np
from tqdm import trange
from monai.inferers import sliding_window_inference
from .metrics import compute_segmentation_metrics, compute_puncta_metrics
from .data import match_labeling, join_connected_puncta
from .metrics import fast_bin_auc, fast_bin_dice
from skimage.measure import label
from sklearn.neighbors import KNeighborsClassifier
from skimage.morphology import binary_opening
import time


def validate(model, loader, loss_fn, slwin_bs=2):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    patch_size = model.patch_size
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
            for l in range(1,n_classes):
                dsc_score.append(fast_bin_dice(labels[l], preds == l))
                auc_score.append(fast_bin_auc(labels[l], preds == l, partial=True))
                if np.isnan(dsc_score[l-1]): dsc_score[l] = 0

            dscs.append(dsc_score)
            aucs.append(auc_score)
            losses.append(loss.item())
            n_elems += 1
            running_dsc += np.mean(dsc_score)
            run_dsc = running_dsc / n_elems
            t.set_postfix(DSC="{:.2f}".format(100 * run_dsc))
            t.update()

    return [100 * np.mean(np.array(dscs)), 100 * np.mean(np.array(aucs)), np.mean(np.array(losses))]


def evaluate(model, loader, slwin_bs=2, compute_metrics=False):
    if model.refinement:
        model_refine = model.refinement
    cfg = model.cfg
    model = model.network
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    patch_size = tuple(map(int, cfg["patch_size"].split('/')))
    out = {}
    with trange(len(loader)) as t:
        for val_data in loader:
            start_time = time.time()
            images, labels = val_data["img"].to(device), val_data["seg"]
            preds = sliding_window_inference(images, patch_size, slwin_bs, model, overlap=0.1, mode='gaussian').cpu()
            preds = preds.argmax(dim=1).squeeze().numpy().astype(np.int8)
            labels = labels.squeeze().numpy().astype(np.int8)
            if str(cfg["model_type"]) == "puncta_detection":
                preds = join_connected_puncta(images.cpu().detach().numpy()[0][0], label(preds, connectivity=3))
                labels = join_connected_puncta(images.cpu().detach().numpy()[0][0], label(labels[1], connectivity=3))
            if labels.shape[0] != preds.shape[0]: labels = labels.argmax(0)
            if bool(cfg["instance_seg"]):
                preds = (preds == 2).astype(int)
                preds = binary_opening(preds)
                preds = label(preds, connectivity=3)
                if bool(cfg["refinement"]):
                    binary_mask = sliding_window_inference(images, patch_size, slwin_bs, model_refine, overlap=0.1, mode='gaussian').cpu()
                    binary_mask = binary_mask.argmax(dim=1).squeeze().numpy().astype(np.int8)
                    preds = train_knn(binary_mask, preds)
                    del binary_mask
            print("Elapsed time:", time.time() - start_time, "seconds")
            if compute_metrics:
                if str(cfg["model_type"]) == "segmentation":
                    preds = match_labeling(labels, preds)
                    out = compute_segmentation_metrics(preds, labels, out)
                elif str(cfg["model_type"]) == "puncta_detection":
                    out = compute_puncta_metrics(preds, labels, out)
            else:
                if len(out) == 0: out['seg'] = []
                out['seg'].append(preds)
            del images
            del preds
            del labels
            t.update()

    return out


def train_knn(borders: np.array, preds: np.array) -> np.array:
    training_samples = []
    borders = borders.astype(bool) & (~preds.astype(bool))
    borders = borders.astype(int)
    preds[borders == 1] = -1
    for l in np.unique(preds)[1:]:
        z_coords, y_coords, x_coords = np.where(preds == l)
        training_samples.append(np.stack([x_coords, y_coords, z_coords, l * np.ones_like(x_coords)], axis=1))

    training_samples = np.vstack(training_samples)
    x_train, y_train = training_samples[:, :3], training_samples[:, 3]

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)

    z_coords, y_coords, x_coords = np.where(borders == 1)
    x_test = np.stack([x_coords, y_coords, z_coords], axis=1)
    y_pred = knn.predict(x_test)

    preds[borders == 1] = 0
    borders[borders == 1] = y_pred.astype(int)

    preds = preds + borders
    return preds