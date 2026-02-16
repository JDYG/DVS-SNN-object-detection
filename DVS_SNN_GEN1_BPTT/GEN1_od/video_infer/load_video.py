import numpy as np
from utils.common import get_list
from utils.psee_toolbox.io.psee_loader import PSEELoader
import torch
from torch.utils import data

class VideoLoader(data.Dataset):
    def __init__(self, fpath_evt_lst = None, fpath_lbl_lst = None, stride_t = None, skip_t=5e5, min_box_diag=30, min_box_side=10,):
        self.stride_t = stride_t
        self.train_duration = 0

        self.min_box_diag = min_box_diag
        self.min_box_side = min_box_side
        self.skip_t = skip_t

        # if fpath_evt_lst end with '.txt'
        if fpath_evt_lst.endswith('.txt'):
            self.list_fpath_evt = get_list(fpath_evt_lst, ext=None)  # return the file names of events in /data/gen1/list/train/events.txt
            self.list_fpath_lbl = get_list(fpath_lbl_lst, ext=None)
        if fpath_evt_lst.endswith('.dat'):
            self.list_fpath_evt = [fpath_evt_lst]  # return the file names of events in /data/gen1/list/train/events.txt
            self.list_fpath_lbl = [fpath_evt_lst[:-7] + '_bbox.npy']

        self.data_len = len(self.list_fpath_evt)


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        event_path = self.list_fpath_evt[index]
        label_path = self.list_fpath_lbl[index]
        # print(f'index: {index}, event_path: {event_path}')
        events_streams, bbox_dict = self.load_video(event_path, label_path, self.stride_t, self.skip_t, min_box_diag=30, min_box_side=10)
        return events_streams, bbox_dict, event_path



    def load_video(self,event_path, label_path, stride_t, skip_t=5e5, min_box_diag=30, min_box_side=10):
        events_video = PSEELoader(event_path)
        box_video = PSEELoader(label_path)
        seek_ts = skip_t

        events_video.seek_time(seek_ts)
        box_video.seek_time(seek_ts)
        total_time = events_video.total_time()
        total_time = 10e6
        self.train_duration = total_time
        events = events_video.load_delta_t(total_time)
        labels = box_video.load_delta_t(total_time)
        events['t'] = events['t'] - skip_t
        labels['t'] = labels['t'] - skip_t

        events, labels = self._bind(events, labels, 0)  # events is a Nx4 array, (t,x,y,p), labels is a Mx6 array,  (t,x_left, y_left, x_right, y_right,class_id)
        event_dict = {'events': events}
        bbox_dict = self._labels2bboxdict(labels)
        bbox_dict = self._filter_small_bboxes(bbox_dict, self.min_box_diag, self.min_box_side)

        events_streams = self.seg_events(event_dict)

        return events_streams, bbox_dict




    # segment the events into strides
    # return a list of dicts, [{'events': torch.Tensor(N1,4)}, {'events': torch.Tensor(N2,4)}, ...]
    def seg_events(self,event_dict):
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
    def _split_by_indices(self, data_array, indices):
        split_indices = np.flatnonzero(indices[1:] - indices[:-1]) + 1
        data_splits = np.split(data_array, split_indices)
        new_indices = np.unique(indices)
        return data_splits, new_indices
    def _bind(self,events, labels, seek_ts):
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

    def _filter_small_bboxes(self, bbox_dict, min_box_diag=30, min_box_side=10):
        bboxes = bbox_dict['bboxes']
        W = bboxes[:,2] - bboxes[:,0]
        H = bboxes[:,3] - bboxes[:,1]
        diag_square = W**2+H**2
        min_side = torch.minimum(W,H)
        mask = torch.logical_and(diag_square >= min_box_diag**2, min_side >= min_box_side).to(bool)
        bbox_dict['valid'] = mask
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

    def _labels2bboxdict(self,  labels):
        if len(labels) == 0:
            gt_times = np.empty([0], dtype=int)
            gt_bboxes = np.empty([0,4], dtype=int)
            gt_labels = np.empty([0], dtype=int)
            valid_mask = np.empty([0], dtype=bool)
        else:
            gt_times = labels[:,0]
            gt_times = gt_times // self.stride_t
            gt_bboxes = labels[:,1:5]
            gt_labels = labels[:,5]
            if labels.shape[1] == 7:
                valid_mask = labels[:,6]
            else:
                valid_mask = np.ones_like(gt_labels).astype(bool)

        bbox_dict = {
            'index': torch.from_numpy(gt_times),
            'bboxes': torch.from_numpy(gt_bboxes),
            'labels': torch.from_numpy(gt_labels),
            'valid': torch.from_numpy(valid_mask).bool(),
        }

        return bbox_dict
    # def _filter_small_bboxes(self, bbox_dict, min_box_diag=30, min_box_side=10):
    #     bboxes = bbox_dict['bboxes']
    #     W = bboxes[:,2] - bboxes[:,0]
    #     H = bboxes[:,3] - bboxes[:,1]
    #     diag_square = W**2+H**2
    #     min_side = torch.minimum(W,H)
    #     mask = torch.logical_and(diag_square >= min_box_diag**2, min_side >= min_box_side).to(bool)
    #     bbox_dict['valid'] = mask
    #     return bbox_dict




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



    @property
    def data(self):
        return self._backet

    def __len__(self):
        return len(self._backet)
