"""Model prediction / inference functions."""

import numpy as np
from tqdm import trange
from monai.inferers import sliding_window_inference
from .metrics import compute_segmentation_metrics, compute_puncta_metrics
from .data import match_labeling
from .metrics import fast_bin_auc, fast_bin_dice
from skimage.measure import label
from sklearn.neighbors import KNeighborsClassifier


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
                if np.isnan(dsc_score[l]): dsc_score[l] = 0

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
            if bool(cfg["instance_seg"]):
                borders = (preds == 1).astype(int)
                preds = (preds == 2).astype(int)
                preds = match_labeling(labels, label(preds))
                if bool(cfg["knn"]):
                    preds = train_KNN(borders, preds)
            if str(cfg["model_type"]) == "segmentation":
                metrics = compute_segmentation_metrics(preds, labels, metrics)
            elif str(cfg["model_type"]) == "puncta_detection":
                metrics = compute_puncta_metrics(preds, labels, metrics)
            del preds
            del labels
            t.update()

    return metrics


def train_KNN(borders: np.array, preds: np.array) -> np.array:
    preds[borders != 0] = -1  # Mask out the border pixels by setting them to -1
    training_samples = []
    for l in np.unique(preds):
        if l == -1:
            continue
        x_coords, y_coords, z_coords = np.where(preds == l)
        training_samples.append(np.stack([x_coords, y_coords, z_coords, l * np.ones_like(x_coords)], axis=1))

    training_samples = np.vstack(training_samples)
    x_train, y_train = training_samples[:, :3], training_samples[:, 3]

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)

    x_test = np.column_stack(np.where(borders))
    y_pred = knn.predict(x_test)

    borders[np.where(borders)] = y_pred.astype(int)

    preds = preds + borders
    return preds