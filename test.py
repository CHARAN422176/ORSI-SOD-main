import torch
import torch.nn.functional as F
import numpy as np
import os, argparse, time, csv
import cv2

from model.CPD_models import JRBM
from model.CPD_ResNet_models import JRBM_ResNet
from data import test_dataset

# ================= Metrics =================

eps = 2.2204e-16

def im2double(im):
    return cv2.normalize(im.astype('float'),
                         None,
                         0.0, 1.0,
                         cv2.NORM_MINMAX)

def parameter():
    p = {}
    p['gtThreshold'] = 0.5
    p['beta'] = np.sqrt(0.3)
    p['thNum'] = 100
    p['thList'] = np.linspace(0, 1, p['thNum'])
    return p

def prCount(gtMask, curSMap, p):
    gtMask = (gtMask >= p['gtThreshold']).astype(np.float32)
    gtInd = np.where(gtMask > 0)
    gtCnt = np.sum(gtMask)

    hitCnt = np.zeros((p['thNum'], 1), np.float32)
    algCnt = np.zeros((p['thNum'], 1), np.float32)

    for k, curTh in enumerate(p['thList']):
        thSMap = (curSMap >= curTh).astype(np.float32)
        hitCnt[k] = np.sum(thSMap[gtInd])
        algCnt[k] = np.sum(thSMap)

    prec = hitCnt / (algCnt + eps)
    recall = hitCnt / (gtCnt + 1e-10)
    return prec, recall

def PR_Curve(preds, gts):
    p = parameter()
    beta = p['beta']
    prec, recall = [], []

    for curMap, curGT in zip(preds, gts):
        if curMap.shape != curGT.shape:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))
        curPrec, curRecall = prCount(curGT, curMap, p)
        prec.append(curPrec)
        recall.append(curRecall)

    prec = np.hstack(prec[:])
    recall = np.hstack(recall[:])

    prec = np.mean(prec, 1)
    recall = np.mean(recall, 1)

    score = (1+beta**2) * prec * recall / (beta**2 * prec + recall + eps)
    return np.max(score)

def MAE_Value(preds, gts):
    MAE = []
    for curMap, curGT in zip(preds, gts):
        if curMap.shape != curGT.shape:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))
        curGT = (curGT >= 0.5).astype(np.float32)
        MAE.append(np.mean(np.abs(curMap - curGT)))
    return np.mean(MAE)

# ---- E-measure ----
def Emeasure(pred, gt):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + eps)
    gt = (gt >= 0.5).astype(np.float32)
    fm = pred - pred.mean()
    gt = gt - gt.mean()
    align = 2 * np.sum(fm * gt) / (np.sum(fm * fm) + np.sum(gt * gt) + eps)
    score = ((align + 1) * (align + 1)) / 4
    return score

def Emeasure_Value(preds, gts):
    scores = []
    for curMap, curGT in zip(preds, gts):
        if curMap.shape != curGT.shape:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))
        scores.append(Emeasure(curMap, curGT))
    return np.mean(scores)

# ---- S-measure ----
def S_object(pred, gt):
    fg = np.sum(pred * gt)
    return 2 * fg / (np.sum(pred) + np.sum(gt) + eps)

def Smeasure(pred, gt):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + eps)
    gt = (gt >= 0.5).astype(np.float32)
    return S_object(pred, gt)

def Smeasure_Value(preds, gts):
    scores = []
    for curMap, curGT in zip(preds, gts):
        if curMap.shape != curGT.shape:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))
        scores.append(Smeasure(curMap, curGT))
    return np.mean(scores)

# ================= Main =================

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_path = '/kaggle/input/'
test_datasets = ['eorssd/']

j = 1
results = []

start_time = time.time()

while j <= 1:
    model = JRBM(32)
    # model = JRBM_ResNet(32)  # If ResNet backbone
    model.load_state_dict(torch.load('/kaggle/input/orsi_sod_pretrain/pytorch/default/2/models/models/eorssd_vgg/model-1'))
    model.cuda()
    model.eval()

    for dataset in test_datasets:
        print(f"Running dataset {dataset}, epoch {j}")

        image_root = dataset_path + dataset + 'test-images/'
        gt_root = dataset_path + dataset + '/test-labels/'

        test_loader = test_dataset(image_root, gt_root, gt_root, gt_root, opt.testsize)

        preds, gts = [], []

        for i in range(test_loader.size):
            image, gt, _, _, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            with torch.no_grad():
                _, _, out3 = model(image)

            res = F.interpolate(out3, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            preds.append(res)
            gts.append(gt)

        mae = MAE_Value(preds, gts)
        fmeasure = PR_Curve(preds, gts)
        smeasure = Smeasure_Value(preds, gts)
        emeasure = Emeasure_Value(preds, gts)

        print(f"Epoch {j} | F: {fmeasure:.4f} | MAE: {mae:.4f} | S: {smeasure:.4f} | E: {emeasure:.4f}")

        results.append([j, fmeasure, mae, smeasure, emeasure])

    j += 1

total_time = time.time() - start_time
print(f"\n✅ Finished all in {total_time:.2f} seconds")

# Save results
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "F-measure", "MAE", "S-measure", "E-measure"])
    writer.writerows(results)
