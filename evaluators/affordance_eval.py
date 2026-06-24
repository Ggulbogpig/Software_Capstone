import os
import logging
import torch
import json
from typing import Any
from common.logger import MetricLogger, SmoothedValue
from common.registry import registry
from common.utils import get_rank


def calculate_mask_iou(mask1, mask2):
    # mask: (h, w)
    intersection = torch.logical_and(mask1, mask2).sum().float()
    union = torch.logical_or(mask1, mask2).sum().float()
    iou = intersection / union
    return iou


def calculate_mask_iou_with_thre(mask1, mask2, thre):
    mask1 = (mask1 >= thre).to(torch.float32)
    intersection = torch.logical_and(mask1, mask2).sum().float()
    union = torch.logical_or(mask1, mask2).sum().float()
    # print("intersection,uninon",intersection,union)
    iou = intersection / (union + 1e-6)
    iou[union == 0] += 1.0
    return iou


def calculate_precision_recall_accuracy(pred, gt):
    pred = pred.squeeze()
    gt = gt.squeeze()
    equal_points = pred == gt
    true_positives = ((pred == 1) & (gt == 1)).sum().item()
    positives_pred = pred.sum().item()
    positives_gt = gt.sum().item()
    precision = true_positives / positives_pred if positives_pred > 0 else 0
    recall = true_positives / positives_gt if positives_gt > 0 else 0
    accuracy = equal_points.float().mean().item()

    return accuracy, precision, recall


@registry.register_evaluator("affordance_acc")
class AffordanceAccEval:
    def __init__(self, name) -> None:
        self.name = name

    def eval_step(self, model, samples, counter) -> Any:
        samples.update({"category": "ref"})
        # output = model.generate(
        #     samples,
        #     num_beams=1,
        #     max_length=30,
        # )
        #answer = output["text"]
        #pred_mask = output["masks"]

        all_masks = []
        all_texts = []

        chunk_size = 8

        batch_size = len(samples["question"])

        # for start in range(0, batch_size, chunk_size):

        #     end = start + chunk_size

        #     sub_samples = {}

        #     for k, v in samples.items():

        #         if isinstance(v, list):
        #             sub_samples[k] = v[start:end]

        #         else:
        #             sub_samples[k] = v[start:end]
            #####추가 특정 오브젝트만 추출#########
        target_id = "f3a7f8198cc50c225f5e789acd4d1122" #컵 - 손잡이 달린
        found_target = False    

        for start in range(0, batch_size, chunk_size):

                end = start + chunk_size

                sub_samples = {}

                for k, v in samples.items():

                    if isinstance(v, list):
                        sub_samples[k] = v[start:end]

                    else:
                        sub_samples[k] = v[start:end]

                chunk_ids = sub_samples["shape_id"]
                print(chunk_ids)

                if target_id not in chunk_ids:
                    continue

                found_target=True

                print("Found Target Chunk")
###################################################
                output = model.generate(
                    sub_samples,
                    num_beams=1,
                    max_length=30,
                )
                if not found_target:
                    return 0, 0, 0, 0, 0, []

                for i in range(len(output["masks"])):

                    if sub_samples["shape_id"][i] != target_id:
                        continue
                    torch.save(
                        {
                            "pred": output["masks"][i].cpu(),
                            "gt": sub_samples["masks"][i].cpu(),
                            "points": sub_samples["points"][i].cpu(),
                            "question": sub_samples["question"][i],
                            "shape_id": sub_samples["shape_id"][i],
                        },
                        f"{target_id}.pt"
                    )
                    print("Found Target")
                    exit()

                # all_masks.extend(output["masks"])

                # all_texts.extend(output["text"])

                # del output

                torch.cuda.empty_cache()
     
                

        pred_mask = all_masks

        answer = all_texts
        

    
        os.makedirs("/data/hbsssssong/mask_outputs", exist_ok=True)


        # for i in range(len(pred_mask)):

        #     if sub_samples["shape_id"][i] != target_id:
        #         continue

        #     torch.save(
        #         {
        #             "pred": pred_mask[i].cpu(),
        #             "gt": sub_samples["masks"][i].cpu(),
        #             "points": sub_samples["points"][i].cpu(),
        #             "question": sub_samples["question"][i],
        #             "shape_id": sub_samples["shape_id"][i],
        #         },
                
        #         #f"mask_outputs/batch_{counter}_sample_{i}.pt"
        #         f"{target_id}.pt"
        #     )
        #     #print(f"mask_outputs/batch_{counter}_sample_{i}.pt")
        #     print("FOUND TARGET!")

        # exit()

        iou, correct_5 = 0.0, 0.0

        pointAcc, pointPrecision, pointRecall = 0.0, 0.0, 0.0

        num = sum([gt_masks.shape[0] for gt_masks in samples["masks"]])

        for idx, (pred, gt) in enumerate(zip(pred_mask, samples["masks"])):
            n_pred = pred.shape[0]
            n_gt = gt.shape[0]

            if n_gt == 0:
                continue

            if n_pred > n_gt:
                pred = pred[:n_gt, ...]

            for mask1, mask2 in zip(pred, gt):
                if mask1.sum() == 0:
                    continue

                _iou = calculate_mask_iou(mask1, mask2)

                _pointAcc, _pointPrecision, _pointRecall = (
                    calculate_precision_recall_accuracy(pred=mask1, gt=mask2)
                )

                correct_5 += _iou > 0.5

                iou += _iou

                pointAcc += _pointAcc

                pointPrecision += _pointPrecision

                pointRecall += _pointRecall


        return (
            correct_5 / num * 100,
            iou / num * 100,
            pointAcc / num * 100,
            pointPrecision / num * 100,
            pointRecall / num * 100,
            [
                {"question": q, "pred": ans, "gt": gt}
                for ans, gt, q in zip(answer, samples["answer"], samples["question"])
            ],
        )

    def __call__(self, model, dataloader, dir, print_freq=100) -> Any:
        logging.info(f"Start evaluating on {self.name}")
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "acc5", SmoothedValue(fmt="global_acc: {global_avg:.6f}")
        )
        metric_logger.add_meter(
            "iou", SmoothedValue(fmt="global_iou: {global_avg:.6f}")
        )
        metric_logger.add_meter(
            "pointAcc", SmoothedValue(fmt="global_pointAcc: {global_avg:.6f}")
        )
        metric_logger.add_meter(
            "pointPrecision",
            SmoothedValue(fmt="global_pointPrecision: {global_avg:.6f}"),
        )
        metric_logger.add_meter(
            "pointRecall", SmoothedValue(fmt="global_pointRecall: {global_avg:.6f}")
        )

        results = []  # Initialize a dictionary to store indicator values
        # Used for debugging, stopping before certain data
        counter = 0

        for samples in metric_logger.log_every(dataloader, print_freq, self.name):
            # print("frequeu",print_freq)
            acc5, iou, pointAcc, pointPrecision, pointRecall, result = self.eval_step(
                model, samples, counter
            )
            results.extend(result)
            metric_logger.update(
                acc5=acc5,
                iou=iou,
                pointAcc=pointAcc,
                pointPrecision=pointPrecision,
                pointRecall=pointRecall,
            )
            counter += 1
   
        result_dir = os.path.join(dir, self.name)
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, f"{get_rank()}.json"), "w") as f:
            json.dump(results, f)

        metric_logger.synchronize_between_processes()
        logging.info(metric_logger.global_avg())
        return {
            "acc5_global_avg": metric_logger.acc5.global_avg,
            "iou_global_avg": metric_logger.iou.global_avg,
            "pointAcc_global_avg": metric_logger.pointAcc.global_avg,
            "pointPrecision_global_avg": metric_logger.pointPrecision.global_avg,
            "pointRecall_global_avg": metric_logger.pointRecall.global_avg,
        }
