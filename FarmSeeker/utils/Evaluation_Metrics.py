import os
import numpy as np
from PIL import Image
import json
# ---------------------------- Evaluator 类（修正版）----------------------------
class Evaluator_all(object):
    def __init__(self, gts,preds,corrects,num_class=2):
        self.num_class = num_class
        self.confusion_matrix_base = np.zeros((num_class, num_class))
        self.confusion_matrix_correct = np.zeros((num_class, num_class))
        self.gts=gts
        self.preds=preds
        self.corrects=corrects

    def _generate_matrix(self, gt_image, pred_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pred_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self):
        for gt_image, pred_image,corr_image in zip(self.gts,self.preds,self.corrects):
            self.confusion_matrix_base += self._generate_matrix(gt_image, pred_image)
            self.confusion_matrix_correct+=self._generate_matrix(gt_image,corr_image)
    def reset(self):
        self.confusion_matrix_base = np.zeros((self.num_class, self.num_class))
        self.confusion_matrix_correct = np.zeros((self.num_class, self.num_class))
    def Pixel_Accuracy(self):
        acc_base = np.diag(self.confusion_matrix_base).sum() / self.confusion_matrix_base.sum()
        acc_corr = np.diag(self.confusion_matrix_correct).sum() / self.confusion_matrix_correct.sum()
        return acc_base,acc_corr
    def Pixel_Accuracy_Class(self):
        acc_per_class_base = np.diag(self.confusion_matrix_base) / (np.sum(self.confusion_matrix_base, axis=1) + 1e-10)
        acc_per_class_corr = np.diag(self.confusion_matrix_correct) / (np.sum(self.confusion_matrix_correct, axis=1) + 1e-10)
        return np.nanmean(acc_per_class_base),np.nanmean(acc_per_class_corr)
    def Mean_Intersection_over_Union(self):
        intersection_base = np.diag(self.confusion_matrix_base)
        intersection_corr = np.diag(self.confusion_matrix_correct)
        union_base = np.sum(self.confusion_matrix_base, axis=1) + np.sum(self.confusion_matrix_base, axis=0) - intersection_base
        union_corr = np.sum(self.confusion_matrix_correct, axis=1) + np.sum(self.confusion_matrix_correct, axis=0) - intersection_corr
        iou_base = intersection_base / (union_base + 1e-10)
        iou_corr = intersection_corr / (union_corr + 1e-10)
        return np.nanmean(iou_base),np.nanmean(iou_corr)
    def Frequency_Weighted_Intersection_over_Union(self):
        freq_base = np.sum(self.confusion_matrix_base, axis=1) / (np.sum(self.confusion_matrix_base) + 1e-10)
        iu_base = np.diag(self.confusion_matrix_base) / (
                    np.sum(self.confusion_matrix_base, axis=1) + np.sum(self.confusion_matrix_base, axis=0) -
                    np.diag(self.confusion_matrix_base) + 1e-10)
        fw_iou_base = (freq_base[freq_base > 0] * iu_base[freq_base > 0]).sum()

        freq_corr = np.sum(self.confusion_matrix_correct, axis=1) / (np.sum(self.confusion_matrix_correct) + 1e-10)
        iu_corr = np.diag(self.confusion_matrix_correct) / (
                    np.sum(self.confusion_matrix_correct, axis=1) + np.sum(self.confusion_matrix_correct, axis=0) -
                    np.diag(self.confusion_matrix_correct) + 1e-10)
        fw_iou_corr = (freq_corr[freq_corr > 0] * iu_corr[freq_corr > 0]).sum()
        return fw_iou_base,fw_iou_corr
    def calculate_iou_f1_recall(self):
        intersection_base = np.diag(self.confusion_matrix_base)
        union_base = np.sum(self.confusion_matrix_base, axis=1) + np.sum(self.confusion_matrix_base, axis=0) - intersection_base
        iou_base = intersection_base / (union_base + 1e-10)
        dice_base = 2 * intersection_base / (np.sum(self.confusion_matrix_base, axis=1) + np.sum(self.confusion_matrix_base, axis=0) + 1e-10)
        recall_base = intersection_base / (np.sum(self.confusion_matrix_base, axis=0) + 1e-10)  # TP / (TP + FN)
        base_iou_f1_recall=[iou_base.mean(), dice_base.mean(), recall_base.mean()]

        intersection_corr = np.diag(self.confusion_matrix_correct)
        union_corr = np.sum(self.confusion_matrix_correct, axis=1) + np.sum(self.confusion_matrix_correct, axis=0) - intersection_corr
        iou_corr = intersection_corr / (union_corr + 1e-10)
        dice_corr = 2 * intersection_corr / (np.sum(self.confusion_matrix_correct, axis=1) + np.sum(self.confusion_matrix_correct, axis=0) + 1e-10)
        recall_corr = intersection_corr / (np.sum(self.confusion_matrix_correct, axis=0) + 1e-10)  # TP / (TP + FN)
        corr_iou_f1_recall=[iou_corr.mean(), dice_corr.mean(), recall_corr.mean()]
        return base_iou_f1_recall,corr_iou_f1_recall
    def calculate_pos1_metrics(self, positive=1):
        cm = self.confusion_matrix_base.astype(np.float64)
        tp = cm[positive, positive]
        fn = cm[positive, :].sum() - tp     # 该类GT被错判
        fp = cm[:, positive].sum() - tp     # 其他类被错判为该类
        tn = cm.sum() - tp - fp - fn
        eps = 1e-10
        iou    = tp / (tp + fp + fn + eps)
        dice   = 2 * tp / (2 * tp + fp + fn + eps)
        recall = tp / (tp + fn + eps)       # TP/(TP+FN)
        acc    = (tp + tn) / (tp + fp + fn + tn + eps)
        base_evaluation=[float(iou), float(dice), float(recall), float(acc)]
        
        cm_corr = self.confusion_matrix_correct.astype(np.float64)
        tp_corr = cm_corr[positive, positive]
        fn_corr = cm_corr[positive, :].sum() - tp_corr     # 该类GT被错判
        fp_corr = cm_corr[:, positive].sum() - tp_corr     # 其他类被错判为该类
        tn_corr = cm_corr.sum() - tp_corr - fp_corr - fn_corr
        eps_corr = 1e-10
        iou_corr    = tp_corr / (tp_corr + fp_corr + fn_corr + eps_corr)
        dice_corr   = 2 * tp_corr / (2 * tp_corr + fp_corr + fn_corr + eps_corr)
        recall_corr = tp_corr / (tp_corr + fn_corr + eps_corr)       # TP/(TP+FN)
        acc_corr    = (tp_corr + tn_corr) / (tp_corr + fp_corr + fn_corr + tn_corr + eps_corr)
        corr_evaluation=[float(iou_corr), float(dice_corr), float(recall_corr), float(acc_corr)]
        return base_evaluation,corr_evaluation
    def compute_correction_accuracy(self,ground_truth, baseline, corrected):
        """计算单张图像的纠正准确率 (CA)"""
        gt = np.asarray(ground_truth)
        bl = np.asarray(baseline)
        cr = np.asarray(corrected)
        if gt.shape != bl.shape or bl.shape != cr.shape:
            raise ValueError(f"形状不匹配: GT{gt.shape}, BL{bl.shape}, CR{cr.shape}")
        modified = (bl != cr)
        tc = (bl != gt) & (cr == gt) & modified  # True Correction
        fa = (bl == gt) & (cr != gt) & modified  # False Alarm
        tc_count = np.sum(tc)
        fa_count = np.sum(fa)
        total_changes = tc_count + fa_count
        # return np.nan if total_changes == 0 else tc_count / total_changes
        return tc_count,fa_count
    def compute_correction_accuracy_batch(self):
       
        tc_all_count=0
        fc_all_count=0
        for gt,pred,corr in zip(self.gts,self.preds,self.corrects):
            tc_count,fa_count = self.compute_correction_accuracy(gt, pred, corr)
            tc_all_count+=tc_count
            fc_all_count+=fa_count
        total_changes=tc_all_count+fc_all_count
        ca=np.nan if total_changes == 0 else  tc_all_count / total_changes
        if np.isnan(ca):
            print(f"该图像分割结果没有修改")
        else:
            print(f"CA = {ca:.3f}")
        # mean_ca = np.nanmean(ca_list) if len(ca_list) > 0 else np.nan
        print(f"\n纠正准确率= {ca:.4f}")
        return ca
def calculate_all_metrics(gts,preds,corrects,save_path):
    all_evalutor=Evaluator_all(gts,preds,corrects)
    all_evalutor.add_batch()
    results=[]
    average_metrics=all_evalutor.calculate_iou_f1_recall()
    post1_metrics=all_evalutor.calculate_pos1_metrics()
    mean_ca=all_evalutor.compute_correction_accuracy_batch()
    results.append({"base":{'mIoU':average_metrics[0][0],"mDice":average_metrics[0][1],"mRecall":average_metrics[0][2],'IoU_cls':post1_metrics[0][0],"Dice_cls":post1_metrics[0][1],"Recall_cls":post1_metrics[0][2],"Acc_cls":post1_metrics[0][3]},
                    "correct":{'mIoU':average_metrics[1][0],"mDice":average_metrics[1][1],"mRecall":average_metrics[1][2],'IoU_cls':post1_metrics[1][0],"Dice_cls":post1_metrics[1][1],"Recall_cls":post1_metrics[1][2],"Acc_cls":post1_metrics[1][3]},
                    "mecca":mean_ca})
    print(results[0])
    with open(save_path,"w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=4)
    return post1_metrics[0][0],post1_metrics[1][0]




