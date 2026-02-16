import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss_Yolox:
    def __init__(self, model, cfg_loss, autobalance=False):
        device = next(model.parameters()).device
        m = model.detection.model[-1]  # Detect() module
        self.strides = m.strides
        nl = len(self.strides)
        self.num_classes = m.num_classes
        self.grids = [torch.zeros(1, device=device)] * nl
        self.n_anchors = 1

        self.iou_loss = IOUloss(reduction="none")
        # self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        # self.bcewithlog_loss_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg_loss['cls_pw'], device=device))
        # self.bcewithlog_loss_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg_loss['obj_pw'], device=device))
        self.bcewithlog_loss_cls = nn.BCEWithLogitsLoss(reduction="none")
        self.bcewithlog_loss_obj = nn.BCEWithLogitsLoss(reduction="none")

        g = cfg_loss['fl_gamma'] # focal loss gamma
        if g > 0:
            self.bcewithlog_loss_cls = FocalLoss(self.bcewithlog_loss_cls, g)
            self.bcewithlog_loss_obj = FocalLoss(self.bcewithlog_loss_obj, g)


    def __call__(self, p, targets_, valid_mask):
        # input p : (bs, 4, 60 ,76), (bs, 4, 30, 38), (bs, 4, 15, 19)
        #           (bs, 1, 60, 76), (bs, 1, 30, 38), (bs, 1, 15, 19)
        #           (bs, nc, 60, 76), (bs, nc, 30, 38), (bs, nc, 15, 19)
        targets = targets_.clone()
        img_w, img_h = 304, 240
         # (cls, cx_n, cy_n, w_n, h_n) - > (cls, cx, xy, w, h)
        targets[:, 2] = targets[:, 2] * img_w
        targets[:, 3] = targets[:, 3] * img_h
        targets[:, 4] = targets[:, 4] * img_w
        targets[:, 5] = targets[:, 5] * img_h

        reg_outputs, obj_outputs, cls_outputs = p # (15,19) , (30, 38), (60, 76)

        # check the stride order
        h0, h1, h2 = reg_outputs[0].shape[2], reg_outputs[1].shape[2], reg_outputs[2].shape[2]
        assert h0*self.strides[0] == h1*self.strides[1] == h2*self.strides[2], "stride order is not correct"

        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for k, stride_this_level in enumerate(self.strides):
            reg_output = reg_outputs[k]
            obj_output = obj_outputs[k]
            cls_output = cls_outputs[k]
            output = torch.cat([reg_output, obj_output, cls_output], 1)  # (B,7,60,76),
            output, grid = self.get_output_and_grid(output, k, stride_this_level, reg_output.type())  # output:(B,4560,7), gird:(1,4560,2)

            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(reg_output))

            outputs.append(output)
        loss, loss_items = self.get_losses(x_shifts, y_shifts, expanded_strides, targets, torch.cat(outputs, 1), reg_output.dtype, valid_mask=valid_mask)
        return loss, loss_items

    @torch.jit.ignore
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid(torch.arange(hsize, device=output.device), torch.arange(wsize, device=output.device), indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize) # (B,1,7,60,76)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, n_ch) # (B,4560,7)
        grid = grid.view(1, -1, 2) # (1,4560,2)
        output[..., :2] = (output[..., :2] + grid) * stride # 表示在最后一个维度，也就channel维度上，取前两个维度的值，也就是xy，然后加上grid，再乘以stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride #exp(bbox_wh)*stride, wh投影到输入图像
        return output, grid

    @torch.jit.ignore
    def get_losses(self, x_shifts, y_shifts, expanded_strides, targets, outputs, dtype, valid_mask=None):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # # calculate targets
        # nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            gt_bboxes_per_image = targets[targets[:,0]==batch_idx,2:]
            valid_mask_per_image = valid_mask[valid_mask[:,0]==batch_idx,1]
            num_gt = gt_bboxes_per_image.shape[0] if valid_mask_per_image.sum() > 0 else 0
            num_gts += num_gt
            if num_gt == 0: # no targets
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_classes = targets[targets[:,0]==batch_idx,1]
                bboxes_preds_per_image = bbox_preds[batch_idx] # [5985,4]

                (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img,) \
                    = self.get_assignments(batch_idx, num_gt, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, obj_preds,)
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss_obj(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss_cls(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls
        # return (loss, reg_weight * loss_iou, loss_obj, loss_cls, num_fg / max(num_gts, 1),)
        return loss, (reg_weight * loss_iou, loss_obj, loss_cls)

    @torch.no_grad()
    @torch.jit.ignore
    def get_assignments(self, batch_idx, num_gt, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, obj_preds, mode="gpu",):

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (F.one_hot(gt_classes.to(torch.int64), self.num_classes).float())
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_
        cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + float(1e6) * (~geometry_relation))

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss


        return (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg,)


    def get_geometry_constraint(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation


    ### SimOTA, 求近似最优解 ###
    # 输入：
    #      cost: 通过回归损失和类别损失计算得到的cost
    #      pair_wise_ious: size为[num_gt,num_in_boxes_anchor]的IoU计算，即所有真实框与预测框的IoU
    #      gt_classes: 一幅图像ground truth标注框的类别编号向量
    #      num_gt: 一幅图像的标注框个数
    #      fg_mask: 根据中心点与目标框初步筛选并集掩码

    # 输出：
    #      num_fg: 标签分配完成后，总共存在的候选框个数, the number of fg is very small, 1 or 2 or 3 or ...
    #      gt_matched_classes: 标签分配后，每列候选框预测目标的编号
    #      pred_ious_this_matching: 由标签分配的mask, 筛选真实框与预测框构成的IoU矩阵对应的IoU值
    #      matched_gt_inds: matrix_matching矩阵，存在候选框的位置idx
    # 这一步目的是进一步过滤掉一些无效的候选框，仅仅对一部分候选框计算loss
    @torch.jit.ignore
    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))  # 设置候选框数量
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k,
                                  dim=1)  # 从前面的pair_wise_ious中，给每个gt bbox，挑选10个iou最大的候选框, shape: [num_gt, 10]
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )  # 选择cost值最低的k个 index
            # matching_matrix[gt_idx][pos_idx] = torch.ones(len(pos_idx), device=matching_matrix.device, dtype=matching_matrix.dtype)
            matching_matrix[gt_idx][pos_idx] = 1  # cost值最低的k个候选框，标记为1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:  # 过滤共用的候选框，如果候选框被多个gt bbox 共用，那么只保留cost值最低的那个gt bbox
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)  # 共用候选框对不同gt bbox 的cost的最小值的index
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1  # 只选择cost值最低的那个预选框 对应的gt bbox
        fg_mask_inboxes = matching_matrix.sum(0) > 0  # 所有被选中的候选框的mask
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

# bboxes_a: the gt bboxes, (N1, 4) because the number of gt maybe >1, such as N1,
# bboxes_b: the predicted bboxes, (N2, 4)
@torch.jit.ignore
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2]) # the intersection top-left shape:(N1,N2,2)
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:]) # the intersection bottom-right shape:(N1,N2,2)
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1) # the area of gt bboxes, shape:(N1,)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1) # the area of predicted bboxes, shape:(N2,)
    else:
        tl_a = bboxes_a[:, :2] - (bboxes_a[:, 2:] * 0.5).long() # xl, yl
        tl_b = bboxes_b[:, :2] - (bboxes_b[:, 2:] * 0.5).long()
        br_a = tl_a + bboxes_a[:, 2:]
        br_b = tl_b + bboxes_b[:, 2:]
        tl = torch.max(tl_a[:,None], tl_b[None,:])
        br = torch.min(br_a[:,None], br_b[None,:])
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2) # tl<br is to determine whether the two bboxes have intersection
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all()), the intersection area of two bboxes, shape:(N1,N2)
    return area_i / (area_a[:, None] + area_b - area_i) # intersection / union


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss