import numpy as np
import torch
from torch.utils import data
from utils.psee_toolbox.io.psee_loader import PSEELoader
from utils.common import get_list


class EventPacketStream(data.Dataset):
    def __init__(self, fpath_evt_lst = None, fpath_lbl_lst = None, train_duration = None, stride_t = None, min_box_diag=30, min_box_side=10, output_type = None, event_transform = None, hight_light_p=None):
        self.stride_t = stride_t
        self.train_duration = train_duration
        # self.list_fpath_evt = get_list(fpath_evt_lst, ext=None)  # return the file names of events in /data/gen1/list/train/events.txt
        # self.list_fpath_lbl = get_list(fpath_lbl_lst, ext=None)
        self.event_transform = event_transform
        self.min_box_diag = min_box_diag
        self.min_box_side = min_box_side
        self.output_type = output_type
        self.hight_light_p = hight_light_p
        self.skip_ts = 0.5e6 # skip the first 0.5s of the video # https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/c09d34a4fb8dbfd2db7081bf5e26078c2aa94fc7/src/io/box_filtering.py#L23


        # if fpath_evt_lst end with '.txt'
        if fpath_evt_lst.endswith('.txt'):
            self.list_fpath_evt = get_list(fpath_evt_lst, ext=None)  # return the file names of events in /data/gen1/list/train/events.txt
            self.list_fpath_lbl = get_list(fpath_lbl_lst, ext=None)
        if fpath_evt_lst.endswith('.dat'):
            self.list_fpath_evt = [fpath_evt_lst]  # return the file names of events in /data/gen1/list/train/events.txt
            self.list_fpath_lbl = [fpath_evt_lst[:-7] + '_bbox.npy']



        self.sampling_timings = []
        for ifile, fname_lbl in enumerate(self.list_fpath_lbl):
            box_timestamps = np.unique(np.load(fname_lbl)['ts']).tolist()
            self.sampling_timings += [(ifile, t) for t in box_timestamps if t >= self.skip_ts+self.train_duration]
            # sampling_timings is a list of tuples, each tuple is (file index, timestamp)
            # file index is the index of the events file in the list_fpath_evt
            # timestamp is the timestamp of the bounding box
        self.total_seq = len(self.sampling_timings)


    def __getitem__(self, index):
        event_path = self.list_fpath_evt[self.sampling_timings[index][0]]
        label_path = self.list_fpath_lbl[self.sampling_timings[index][0]]
        events_video = PSEELoader(event_path)
        box_video = PSEELoader(label_path)
        curr_ts = self.sampling_timings[index][1]+1
        seek_ts = int(curr_ts - self.train_duration)
        if seek_ts < 0:
            pass

        events_video.seek_time(seek_ts)
        box_video.seek_time(seek_ts)
        events = events_video.load_delta_t(self.train_duration)
        labels = box_video.load_delta_t(self.train_duration)

        # print(f'index: {index}, curr_ts: {curr_ts}, event_path: {event_path}')
        if len(events['t']) <= 1e3: # if the number of events is too small, return None
            # print(f'****Warning: the number of events is too small, {len(events["t"])}****index: {index}, curr_ts: {curr_ts}, event_path: {event_path}')
            events_streams = [{'events': torch.tensor([[0,0,0,0]])}] * int(self.train_duration // self.stride_t)
            bbox_dict = {'bboxes': torch.tensor([[0.0,0.0,0.0,0.0]]), 'valid': torch.tensor([False]).bool(), 'labels': torch.tensor([0]), 'times': torch.tensor([0])}
            # return events_streams, bbox_dict
            return events_streams, bbox_dict, curr_ts, event_path

        max_event_num = int((self.train_duration // 5e3) * 1e4)
        if len(events['t']) > max_event_num: # 如果数量过多，在dataset的getitem中会进行下采样，变稀疏
            # print(f'****Warning: the number of events is too large, {len(events["t"])}****index: {index}, curr_ts: {curr_ts}, event_path: {event_path}')
            sample_index = np.random.randint(0, len(events['t']), int(max_event_num))
            sample_index = np.sort(sample_index)
            events = events[sample_index]
        # 如果training duration内的bounding box的时间戳不一样（横跨多个gt timestamp），就取最后一个timestamp的bounding box
        if len(np.unique(labels['t'])) != 1: 
            last_ts = labels['t'][-1]
            labels = labels[labels['t'] == last_ts]

        assert len(np.unique(labels['t'])) == 1 # make sure that all the bounding boxes have the same timestamp

        events, labels = self._bind(events, labels, seek_ts) # events is a Nx4 array, (t,x,y,p), labels is a Mx6 array,  (t,x_left, y_left, x_right, y_right,class_id)
        event_dict = {'events': events}
        bbox_dict = self._labels2bboxdict(labels)

        image_meta = {'width': 304, 'height': 240}
        if self.event_transform is not None: # the bbox fed into event_transform is (x_left, y_left, x_right, y_right)
            event_dict, bbox_dict, image_meta = self.event_transform(event_dict, bbox_dict, image_meta, types=['event', 'bbox', 'meta'])
            # the bbox out from the event_transform is (x_left, y_left, x_right, y_right)

        if self.hight_light_p is not None: # hight light events in the gt bbox
            event_dict, bbox_dict, image_meta = hight_light_events_in_bbox(event_dict, bbox_dict, image_meta, p=self.hight_light_p)

        bbox_dict = self._filter_small_bboxes(bbox_dict, self.min_box_diag, self.min_box_side)
        # bbox_dict = self._filter_early_bboxes(bbox_dict, base_time=None, skip_ts=self.skip_ts)
        # bbox_dict {'bboxes': Tensor([x,y,w,h]),'ignore_mask': Tensor(True/False), 'labels': Tensor([class_id]), 'times': Tensor([t])}

        events_streams = self.seg_events(event_dict) # events_streams is a list of dicts, [{'events': torch.Tensor(N1,4)}, {'events': torch.Tensor(N2,4)}, ...], the polarity of events is -1/1
        
        # return events_streams, bbox_dict, curr_ts, event_path, image_meta
        # print(event_path)
        return events_streams, bbox_dict, curr_ts, event_path
        #  events_streams is a list of dicts, [{'events': torch.Tensor(N1,4)}, {'events': torch.Tensor(N2,4)}, ...], the polarity of events is -1/1
        # bbox_dict {'times', 'labels', 'ignore_mask': False, 'bboxes': Tensor([xl,yl, xr, yr])}


    def __len__(self):
        return self.total_seq

    # segment the events into strides
    # return a list of dicts, [{'events': torch.Tensor(N1,4)}, {'events': torch.Tensor(N2,4)}, ...]
    def seg_events(self, event_dict):
        events = event_dict['events'] # events is a Nx4 array, (t,x,y,p)
        num_frames = int(self.train_duration // self.stride_t)
        times_evt = events[:, 0].astype(np.int32)
        indices = (times_evt // self.stride_t).astype(np.int32)
        event_splits, segment_indices_evt = self._split_by_indices(events, indices)
        backet_evt = DataBacket(num=num_frames)
        for event_data, seg_idx in zip(event_splits, segment_indices_evt):
            if seg_idx < 0 or seg_idx >= num_frames:
                continue
            backet_evt.append(seg_idx, event_data)
        event_streams = backet_evt.concat(axis=0, dtype=np.float32)  # Tensor (L, 4)
        data_streams = [{'events': torch.from_numpy(evt)} for evt in event_streams]

        return data_streams


    def _bind(self, events, labels, seek_ts):
        # input labels is a Mx6 array, (t,x_left, y_left, w, h,class_id)
        # output events: Nx4 array of events, (t,x,y,p)
        # output labels: Mx6 array of labels, (t,x_left, y_left, x_right, y_right,class_id)
        events['t'] -= seek_ts
        # events['p'] = events['p'] * 2 - 1  # convert 0 or 1 to -1 or 1
        p_evt = events['p'].astype(np.int32) * 2 - 1
        t_evt, x_evt, y_evt = events['t'], events['x'], events['y']
        events = np.stack([t_evt, x_evt, y_evt, p_evt], axis=-1).astype(np.int32)

        # labels = self._filter_invalid_bboxes(labels)
        # labels['t'] = self.train_duration - 1
        t_lbl, x_lbl, y_lbl, w_lbl, h_lbl, c_lbl = labels['t'], labels['x'], labels['y'], labels['w'], labels['h'], labels['class_id']

        labels = np.stack([t_lbl, x_lbl, y_lbl, w_lbl, h_lbl, c_lbl], axis=-1).astype(np.int32)
        labels = self._xywh2xyxy(labels) # x_left, y_left, w, h -> x_left, y_left, x_right, y_right
        # labels[:,1:5] = self.xyxy2xywhn(labels[:,1:5], w=304, h=240) # x_left, y_left, h, w -> normailzed cx, cy, w, h
        return events, labels


    def _split_by_indices(self, data_array, indices):
        split_indices = np.flatnonzero(indices[1:] - indices[:-1]) + 1
        data_splits = np.split(data_array, split_indices)
        new_indices = np.unique(indices)
        return data_splits, new_indices

    # def _filter_invalid_bboxes(self, labels):
    #     if 'invalid' not in labels.dtype.fields:
    #         return labels
    #     mask = np.logical_not(labels['invalid'])
    #     labels = labels[mask]
    #     return labels



    def _filter_small_bboxes(self, bbox_dict, min_box_diag=30, min_box_side=10):
        bboxes = bbox_dict['bboxes']
        W = bboxes[:,2] - bboxes[:,0]
        H = bboxes[:,3] - bboxes[:,1]
        diag_square = W**2+H**2
        min_side = torch.minimum(W,H)
        mask = torch.logical_and(diag_square >= min_box_diag**2, min_side >= min_box_side).to(bool)
        bbox_dict['valid'] = mask
        return bbox_dict

    def _filter_early_bboxes(self, bbox_dict, base_time=None, skip_ts=0):
        times_lbl = bbox_dict['times']
        valid_mask = times_lbl >= skip_ts
        bbox_dict['valid'] = valid_mask
        return bbox_dict

    def _xywh2xyxy(self, labels, W=304, H=240, clip=True):
        # (t, x_left, y_left, w, h, class) -> (t, x_left, y_left, x_right, y_right, class)
        labels = labels.copy()
        labels[:,3] += labels[:,1]
        labels[:,4] += labels[:,2]

        if clip:
            if isinstance(labels, torch.Tensor):  # faster individually
                labels[..., 1].clamp_(0, W)  # x1
                labels[..., 2].clamp_(0, H)  # y1
                labels[..., 3].clamp_(0, W)  # x2
                labels[..., 4].clamp_(0, H)  # y2
            else:  # np.array (faster grouped)
                labels[..., [1, 3]] = labels[..., [1, 3]].clip(0, W)  # x1, x2
                labels[..., [2, 4]] = labels[..., [2, 4]].clip(0, H)  # y1, y2
        return labels

    # def xyxy2xywhn(self, x, w=304, h=240):
    #     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    #     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    #     y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    #     y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    #     y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    #     y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    #     return y

    def _labels2bboxdict(self, labels):
        if len(labels) == 0:
            gt_times = np.empty([0], dtype=int)
            gt_bboxes = np.empty([0,4], dtype=int)
            gt_labels = np.empty([0], dtype=int)
            valid_mask = np.empty([0], dtype=bool)
        else:
            gt_times = labels[:,0]
            gt_bboxes = labels[:,1:5]
            gt_labels = labels[:,5]
            if labels.shape[1] == 7:
                valid_mask = labels[:,6]
            else:
                valid_mask = np.ones_like(gt_labels).astype(bool)

        bbox_dict = {
            'times': torch.from_numpy(gt_times),
            'bboxes': torch.from_numpy(gt_bboxes),
            'labels': torch.from_numpy(gt_labels),
            'valid': torch.from_numpy(valid_mask).bool(),
        }

        return bbox_dict

    def _bboxdict2labels(self, bbox_dict):
        gt_times = bbox_dict['times']
        gt_bboxes = bbox_dict['bboxes']
        gt_labels = bbox_dict['labels']
        ignore_mask = bbox_dict['ignore_mask']
        labels = torch.cat([gt_times[:,None], gt_bboxes, gt_labels[:,None], ignore_mask[:,None]], dim=-1)
        return labels.numpy()



# hight light events in the gt bbox
# 的作用是只保留gt bounding box内的events
# slack  用于在边界框周围创建一个稍微宽松的区域
# 这样做的目的是为了一开始的时候能让网络的关注点在有意义的events上
# 后面正式训练的时候需要删掉这个操作 

def hight_light_events_in_bbox(event_dict, bbox_dict, image_meta, p):
    # p is the probability
    if bbox_dict['bboxes'].shape[0] == 0:
        return event_dict, bbox_dict, image_meta
    if np.random.uniform(0,1) <= p:
        img_h, img_w = image_meta['height'], image_meta['width']
        slack = 0.1
        valid = []


        for bbox in bbox_dict['bboxes']:
            xl, yl, xr, yr = bbox.cpu().numpy()
            region_h = yr - yl
            region_w = xr - xl
            xl_slack = max(0, xl - region_w * slack)   # the left boundary of the hightlight region
            yl_slack = max(0, yl - region_h * slack)
            xr_slack = min(img_w, xr + region_w * slack)   # the right boundary of the hightlight region
            yr_slack = min(img_h, yr + region_h * slack)

            events = event_dict['events']
            valid_temp = (events[:,1] >= xl_slack) & (events[:,2] >= yl_slack) & (events[:,1] <= xr_slack) & (events[:,2] <= yr_slack)
            valid.append(valid_temp)

        valid = np.stack(valid, axis=0)
        valid = np.any(valid, axis=0)
        output = {}
        output['events'] = event_dict['events'][valid]
        return output, bbox_dict, image_meta

    else:
        return event_dict, bbox_dict, image_meta

# A special data type for events streams
# when a stride does not have any events, if we use traditional 'append' in the python, the events after the empty stride will be moved forward and occupy the empty stride
# for example, if the events are 1,2,3,[],[],6,7
# the traditional 'append' will give 1,2,3,6,7
# but we want 1,2,3,[],[],6,7
# https://github.com/hamarh/HMNet_pth/blob/main/hmnet/dataset/gen1.py
class DataBacket(object):
    def __init__(self, num=1):
        self._backet = [ list() for _ in range(num) ]

    def append(self, idx, data):
        if idx >= len(self._backet):
            num_append = idx - len(self._backet) + 1
            blank_backets = [ list() for _ in range(num_append) ]
            self._backet += blank_backets
        self._backet[idx].append(data)

    def _backet_dtype(self):
        for contents in self._backet:
            if len(contents) > 0:
                return contents[0].dtype

    def concat(self, axis, dtype=None):
        if dtype is None:
            dtype = self._backet_dtype()
        output = []
        for contents in self._backet:
            if len(contents) == 0:
                output.append(np.array([], dtype=dtype))
            else:
                output.append(np.concatenate(contents, axis=axis))
        return output

    def stack(self, axis=0):
        dtype = self._backet_dtype()
        output = []
        for contents in self._backet:
            if len(contents) == 0:
                output.append(np.array([], dtype=dtype))
            else:
                output.append(np.stack(contents, axis=axis))
        return output

    def latest(self):
        dtype = self._backet_dtype()
        output = []
        for contents in self._backet:
            if len(contents) == 0:
                output.append(np.array([], dtype=dtype))
            else:
                output.append(contents[-1])
        return output

    def expand(self, length):
        if length > len(self._backet):
            num_append = length - len(self._backet)
            blank_backets = [ list() for _ in range(num_append) ]
            self._backet += blank_backets

    def pad_blank_backet(self, direction='forward'):
        if direction == 'forward':
            self._pad_forward()
            self._pad_backward()
        elif direction == 'backward':
            self._pad_backward()
            self._pad_forward()

    def blank_backet_as_none(self):
        for i in range(len(self._backet)):
            if len(self._backet[i]) == 0:
                self._backet[i] = None

    def _pad_forward(self):
        pad = None
        for i, backet in enumerate(self._backet):
            if len(backet) > 0:
                pad = backet
            elif pad is not None:
                self._backet[i] = copy.deepcopy(pad)
            else:
                pass

    def _pad_backward(self):
        pad = None
        for i, backet in enumerate(reversed(self._backet)):
            if len(backet) > 0:
                pad = backet
            elif pad is not None:
                self._backet[len(self._backet)-1-i] = copy.deepcopy(pad)
            else:
                pass

    @property
    def data(self):
        return self._backet

    def __len__(self):
        return len(self._backet)