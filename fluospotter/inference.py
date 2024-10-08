"""Model prediction / inference functions."""
import pdb
from tqdm import trange
from monai.inferers import sliding_window_inference
from .metrics import compute_segmentation_metrics, compute_puncta_metrics
from .data import match_labeling, join_connected_puncta
from .metrics import fast_bin_auc, fast_bin_dice
import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import erosion, dilation, disk, binary_opening
from skimage.measure import label, find_contours
from skimage.draw import polygon


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
            for l in range(1, n_classes):
                dsc_score.append(fast_bin_dice(labels[l], preds == l))
                auc_score.append(fast_bin_auc(labels[l], preds == l, partial=True))
                if np.isnan(dsc_score[l - 1]): dsc_score[l] = 0

            dscs.append(dsc_score)
            aucs.append(auc_score)
            losses.append(loss.item())
            n_elems += 1
            running_dsc += np.mean(dsc_score)
            run_dsc = running_dsc / n_elems
            t.set_postfix(DSC="{:.2f}".format(100 * run_dsc))
            t.update()

    return [100 * np.mean(np.array(dscs)), 100 * np.mean(np.array(aucs)), np.mean(np.array(losses))]


def evaluate(model, loader, compute_metrics=False):
    if model.cfg["model_type"] == "segmentation":
        out = evaluate_seg(model, loader, compute_metrics=compute_metrics)
    elif model.cfg["model_type"] == "puncta_detection":
        out = evaluate_puncta(model, loader, compute_metrics=compute_metrics)
    return out


def evaluate_puncta(model, loader, slwin_bs=2, compute_metrics=False):
    model_refine = model.refinement if model.refinement else None
    cfg = model.cfg
    model = model.network
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    patch_size = tuple(map(int, cfg["patch_size"].split('/')))
    out = {}
    with trange(len(loader)) as t:
        for val_data in loader:
            images = val_data["img"].to(device)
            labels = val_data["seg"] if 'seg' in val_data else None
            preds = sliding_window_inference(images, patch_size, slwin_bs, model, overlap=0.1, mode='gaussian').cpu()
            preds = preds.argmax(dim=1).squeeze().numpy().astype(np.int8)
            if labels is not None: labels = labels.squeeze().numpy().astype(np.int8)
            preds = join_connected_puncta(images.cpu().detach().numpy()[0][0], label(preds, connectivity=3))
            if labels is not None: labels = join_connected_puncta(images.cpu().detach().numpy()[0][0],
                                                                  label(labels[1], connectivity=3))
            if bool(cfg["overlapping_puncta"]) and model_refine:
                positions = np.argwhere(preds == 1)
                crop_size = 14
                images = images.squeeze(axis=(0, 1))
                cropped_volume = np.zeros((positions.shape[0], crop_size, crop_size))
                for i in range(positions.shape[0]):
                    z, x, y = positions[i]
                    x_start = max(0, x - crop_size // 2)
                    y_start = max(0, y - crop_size // 2)
                    x_end = min(images.shape[1], x + crop_size // 2)
                    y_end = min(images.shape[2], y + crop_size // 2)
                    if x_end - x_start < crop_size:
                        x_start = max(0, x_end - crop_size)
                        x_end = min(images.shape[1], x_start + crop_size)
                    if y_end - y_start < crop_size:
                        y_start = max(0, y_end - crop_size)
                        y_end = min(images.shape[2], y_start + crop_size)
                    cropped_volume[i] = images[z, x_start:x_end, y_start:y_end]

                cropped_volume = torch.tensor(cropped_volume, dtype=torch.float32).unsqueeze(1)
                mean = cropped_volume.mean(dim=(1, 2), keepdim=True)
                std = cropped_volume.std(dim=(1, 2), keepdim=True)
                cropped_volume = (cropped_volume - mean) / std
                refined_preds = model_refine(cropped_volume)
                overlapping_peak = refined_preds[0].softmax(dim=1).detach().cpu().numpy()
                overlapping_peak = np.argmax(overlapping_peak, axis=1)
                corrected_pos = 13 * torch.nn.functional.sigmoid(refined_preds[1])
                corrected_pos = np.round(corrected_pos.detach().cpu().numpy()).astype(int)
                for i in range(positions.shape[0]):
                    z, x, y = positions[i, 0], positions[i, 1], positions[i, 2]
                    if overlapping_peak[i]:
                        x_start = max(0, x - crop_size // 2)
                        y_start = max(0, y - crop_size // 2)
                        x_end = min(images.shape[1], x + crop_size // 2)
                        y_end = min(images.shape[2], y + crop_size // 2)
                        if x_end - x_start < crop_size: x_start = max(0, x_end - crop_size)
                        if y_end - y_start < crop_size: y_start = max(0, y_end - crop_size)
                        preds[z, x, y] = 0
                        preds[z, x_start + corrected_pos[i][0], y_start + corrected_pos[i][1]] = 1
                        preds[z, x_start + corrected_pos[i][2], y_start + corrected_pos[i][3]] = 1

            if compute_metrics and labels is not None:
                out = compute_puncta_metrics(preds, labels, out)
            else:
                if len(out) == 0:
                    out['puncta'] = []
                    if bool(cfg["overlapping_puncta"]) and model_refine: out['overlapping_puncta'] = []
                positions = np.argwhere(preds == 1)
                out['puncta'].append(positions)
                if bool(cfg["overlapping_puncta"]) and model_refine: out['overlapping_puncta'].append(overlapping_peak)
            del images
            del preds
            del labels
            t.update()
    return out


def gaussian_window(size, sigma=1):
    """Create a 2D Gaussian window."""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)


def evaluate_seg(model, loader, slwin_bs=2, compute_metrics=False, overlap=0.2):
    """
    Evaluate the segmentation model and optionally combine chunked predictions into a single volume.

    Parameters:
    - model: PyTorch model for segmentation.
    - loader: DataLoader for test data.
    - slwin_bs: Batch size for sliding window inference.
    - compute_metrics: Boolean, whether to compute metrics or not.
    - overlap: Float, overlap for sliding window inference.

    Returns:
    - out: Dictionary containing combined segmentation results and optional metrics.
    """
    cfg = model.cfg
    model = model.network
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    patch_size = tuple(map(int, cfg["patch_size"].split('/')))
    im_size = tuple(map(int, cfg["im_size"].split('/')))
    combined_seg = (im_size[1] > 256) or (im_size[2] > 256)
    out = {}

    if combined_seg:
        factor, volume_shape = np.sqrt(loader.dataset[0]['img'].shape[0]), loader.dataset[0]['img'].shape[
                                                                           1:]
        combined_pred = np.zeros((volume_shape[0], im_size[1], im_size[2]),
                                 dtype=np.float32)
        weight_map = np.zeros((volume_shape[0], im_size[1], im_size[2]),
                              dtype=np.float32)

        y_steps = np.ceil((im_size[1] - volume_shape[1]) / (volume_shape[1] * (1 - overlap))).astype(int) + 1
        x_steps = np.ceil((im_size[2] - volume_shape[2]) / (volume_shape[2] * (1 - overlap))).astype(int) + 1
        pad_y, pad_x = int(volume_shape[1] * 0.05 / 2), int(volume_shape[2] * 0.05 / 2)
        gaussian_weights = gaussian_window(volume_shape[1], sigma=volume_shape[1] // 6)

        for _, val_data in enumerate(loader):
            images = val_data["img"].to(device)
            combined_pred.fill(0)
            weight_map.fill(0)

            with trange(images.shape[1]) as t:
                for n in range(images.shape[1]):
                    volume = F.pad(images[:, n], pad=(pad_x, pad_x, pad_y, pad_y, 0, 0), mode='reflect').unsqueeze(0)
                    #volume = images[:,n].unsqueeze(0)
                    preds = sliding_window_inference(volume, patch_size, slwin_bs, model, overlap=overlap,
                                                     mode='gaussian').cpu()
                    preds = preds.argmax(dim=1).squeeze().numpy().astype(np.int8)

                    y_idx = (n // x_steps) % y_steps
                    x_idx = n % x_steps
                    y_start = int(y_idx * volume_shape[1] * (1 - overlap))
                    x_start = int(x_idx * volume_shape[2] * (1 - overlap))
                    y_end = min(y_start + volume_shape[1], im_size[1])
                    x_end = min(x_start + volume_shape[2], im_size[2])

                    combined_pred[:, y_start:y_end, x_start:x_end] += preds[:, :(y_end - y_start),
                                                                      :(x_end - x_start)] * gaussian_weights[
                                                                                            :(y_end - y_start),
                                                                                            :(x_end - x_start)]
                    weight_map[:, y_start:y_end, x_start:x_end] += gaussian_weights[:(y_end - y_start),
                                                                   :(x_end - x_start)]
                    t.update()
                    del preds
                del images

                if bool(cfg["instance_seg"]):
                    preds = (preds == 2).astype(int)
                    preds = binary_opening(preds)
                    preds = label(preds, connectivity=3)

                if len(out) == 0: out['seg'] = []
                combined_pred = np.round(combined_pred / np.maximum(weight_map, 1e-6)).astype(np.int8)
                out['seg'].append(combined_pred)
    else:
        for _, val_data in enumerate(loader):
            images = val_data["img"].to(device)
            labels = val_data["seg"] if 'seg' in val_data else None

            preds = sliding_window_inference(images, patch_size, slwin_bs, model, overlap=overlap,
                                             mode='gaussian').cpu()
            preds = preds.argmax(dim=1).squeeze().numpy().astype(np.int8)

            del images
            if labels is not None:
                labels = labels.squeeze().numpy().astype(np.int8)
                if labels.shape[0] != preds.shape[0]:
                    labels = labels.argmax(0)

            if bool(cfg["instance_seg"]):
                preds = instance_segmentation(preds)

            if compute_metrics and labels is not None:
                preds = match_labeling(labels, preds)
                out = compute_segmentation_metrics(preds, labels, out)
            else:
                if len(out) == 0: out['seg'] = []
                out['seg'].append(preds)
            del preds
            del labels
    return out


def instance_segmentation(mask):
    borders = np.zeros(mask.shape)
    for z in range(borders.shape[0]):
        contours = find_contours(mask[z], 1)
        binary_mask = (mask[z] == 1).astype(int)
        binary_mask = dilation(binary_mask, disk(2))
        x, y = np.where(binary_mask)
        mask_points = set(zip(x, y))

        # Loop through each detected contour
        for contour in contours:
            # Check matching points
            contour_points = set(map(tuple, np.round(contour).astype(int)))

            # Calculate the number of matching points
            matching_points = contour_points & mask_points
            matching_ratio = len(matching_points) / len(contour_points)

            if matching_ratio > 0.8:
                # Save the contour points in the borders array
                contour_int = np.round(contour).astype(int)
                borders[z, contour_int[:, 0], contour_int[:, 1]] = 1

                # Fill the interior of the contour with value 2
                rr, cc = polygon(contour[:, 0], contour[:, 1], binary_mask.shape)

                # Update only the interior pixels where borders are not already 1
                borders[z, rr, cc] = np.where(borders[z, rr, cc] != 1, 2, borders[z, rr, cc])

    instance_mask = label(borders == 2)
    return instance_mask
