import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import os, argparse
import cv2

from model.CPD_models import JRBM
from model.CPD_ResNet_models import JRBM_ResNet
from data import test_dataset
import glob2

eps = 2.2204e-16


# ----------------- Utility -----------------
def parameter():
    p = {}
    p['gtThreshold'] = 0.5
    p['beta'] = np.sqrt(0.3)
    p['thNum'] = 100
    p['thList'] = np.linspace(0, 1, p['thNum'])
    return p


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


# ----------------- PR Curve & MAE -----------------
def prCount(gtMask, curSMap, p):
    gtH, gtW = gtMask.shape[0:2]
    algH, algW = curSMap.shape[0:2]

    if gtH != algH or gtW != algW:
        curSMap = cv2.resize(curSMap, (gtW, gtH))

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


def PR_Curve(resDir, gtDir):
    p = parameter()
    beta = p['beta']
    gtImgs = glob2.iglob(gtDir + '/*.png')

    prec = []
    recall = []
    for gtName in gtImgs:
        dir, name = os.path.split(gtName)
        mapName = os.path.join(resDir, name[:-4] + '.png')

        curMap = im2double(cv2.imread(mapName, cv2.IMREAD_GRAYSCALE))
        curGT = im2double(cv2.imread(gtName, cv2.IMREAD_GRAYSCALE))
        if curMap.shape != curGT.shape:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))

        curPrec, curRecall = prCount(curGT, curMap, p)
        prec.append(curPrec)
        recall.append(curRecall)

    if len(prec) == 0:
        return {'prec': [], 'recall': [], 'curScore': 0, 'curTh': 0, 'fscore': []}

    prec = np.hstack(prec[:])
    recall = np.hstack(recall[:])

    prec = np.mean(prec, 1)
    recall = np.mean(recall, 1)

    score = (1 + beta**2) * prec * recall / (beta**2 * prec + recall + eps)
    curTh = np.argmax(score)
    curScore = np.max(score)

    return {'prec': prec, 'recall': recall, 'curScore': curScore,
            'curTh': curTh, 'fscore': score}


def MAE_Value(resDir, gtDir):
    p = parameter()
    gtThreshold = p['gtThreshold']

    gtImgs = glob2.iglob(gtDir + '/*.png')
    MAE = []

    for gtName in gtImgs:
        dir, name = os.path.split(gtName)
        mapName = os.path.join(resDir, name[:-4] + '.png')

        if not os.path.exists(mapName):
            mapName = mapName.replace('.png', '.jpg')
        if not os.path.exists(mapName):
            mapName = mapName.replace('.jpg', '.bmp')

        curMap = im2double(cv2.imread(mapName, cv2.IMREAD_GRAYSCALE))
        curGT = im2double(cv2.imread(gtName, cv2.IMREAD_GRAYSCALE))
        curGT = (curGT >= gtThreshold).astype(np.float32)

        if curMap.shape != curGT.shape:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[0]))

        diff = np.abs(curMap - curGT)
        MAE.append(np.mean(diff))

    return np.mean(MAE) if MAE else 0


# ----------------- S-measure -----------------
def S_measure(pred, gt):
    y = np.mean(gt)
    if y == 0:
        return 1.0 - np.mean(pred)
    elif y == 1:
        return np.mean(pred)

    x = np.mean(pred)
    alpha = 0.5
    Q = alpha * object_similarity(pred, gt) + (1 - alpha) * region_similarity(pred, gt)
    return Q


def object_similarity(pred, gt):
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)
    o_fg = np.mean(pred[gt == 1]) if np.sum(gt) > 0 else 0
    o_bg = np.mean((1 - pred)[gt == 0]) if np.sum(1 - gt) > 0 else 0
    return 0.5 * (o_fg + o_bg)


def region_similarity(pred, gt):
    x, y = center_of_mass(gt)
    gt1, gt2, gt3, gt4 = divide_quarters(gt, x, y)
    pred1, pred2, pred3, pred4 = divide_quarters(pred, x, y)
    w1, w2, w3, w4 = [np.sum(g) / (np.sum(gt) + eps) for g in [gt1, gt2, gt3, gt4]]
    return w1 * ssim(pred1, gt1) + w2 * ssim(pred2, gt2) + w3 * ssim(pred3, gt3) + w4 * ssim(pred4, gt4)


def center_of_mass(gt):
    y, x = np.where(gt == 1)
    if len(y) == 0:
        return gt.shape[0] // 2, gt.shape[1] // 2
    return int(np.mean(y)), int(np.mean(x))


def divide_quarters(mat, x, y):
    return mat[:x, :y], mat[:x, y:], mat[x:, :y], mat[x:, y:]


def ssim(pred, gt):
    N = pred.size
    mean_x, mean_y = np.mean(pred), np.mean(gt)
    var_x, var_y = np.var(pred), np.var(gt)
    cov_xy = np.mean((pred - mean_x) * (gt - mean_y))
    return ((2 * mean_x * mean_y + eps) * (2 * cov_xy + eps)) / ((mean_x**2 + mean_y**2 + eps) * (var_x + var_y + eps))


# ----------------- E-measure -----------------
def E_measure(pred, gt):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = (gt >= 0.5).astype(np.float32)
    return enhanced_alignment_measure(pred, gt)


def enhanced_alignment_measure(pred, gt):
    numerator = 4 * pred * gt
    denominator = pred**2 + gt**2 + eps
    align_map = numerator / (denominator + eps)
    return np.mean(align_map)


# ----------------- Main -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_path = '/kaggle/input/'
test_datasets = ['eorssd/']

j = 1
while j <= 1:
    model = JRBM(32)
    # model = JRBM_ResNet(32)

    model.load_state_dict(torch.load('/kaggle/input/orsi_sod_pretrain/pytorch/default/2/models/models/eorssd_vgg/model-1'))
    model.cuda()
    model.eval()

    for dataset in test_datasets:
        save_pre = './results_vgg_ors/test_' + str(j) + '/'
        if not os.path.exists(save_pre):
            os.makedirs(save_pre)
        print('j=', j, 'save_path:', save_pre)

        image_root = dataset_path + dataset + 'test-images/'
        gt_root = dataset_path + dataset + 'test-labels/'
        edge_root = gt_root
        gt_back_root = gt_root
        test_loader = test_dataset(image_root, gt_root, edge_root, gt_back_root, opt.testsize)

        # Inference
        for i in range(test_loader.size):
            image, gt, edge, gt_back, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            out1, out2, out3 = model(image)

            res = F.interpolate(out3, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            cv2.imwrite(save_pre + name, res * 255)

        # Evaluation
        gtDir = gt_root
        mae = MAE_Value(save_pre, gtDir)
        pr = PR_Curve(save_pre, gtDir)
        FMeasureF = pr['curScore']

        s_scores, e_scores = [], []
        gtImgs = glob2.iglob(gtDir + '/*.png')
        for gtName in gtImgs:
            dir, name = os.path.split(gtName)
            mapName = os.path.join(save_pre, name[:-4] + '.png')

            curMap = im2double(cv2.imread(mapName, cv2.IMREAD_GRAYSCALE))
            curGT = im2double(cv2.imread(gtName, cv2.IMREAD_GRAYSCALE))
            curGT = (curGT >= 0.5).astype(np.float32)

            s_scores.append(S_measure(curMap, curGT))
            e_scores.append(E_measure(curMap, curGT))

        SMeasureF = np.mean(s_scores) if s_scores else 0
        EMeasureF = np.mean(e_scores) if e_scores else 0

        print('epoch:', j,
              'F-measure:', FMeasureF,
              'MAE:', mae,
              'S-measure:', SMeasureF,
              'E-measure:', EMeasureF)

        j += 1
