import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

from PIL import Image
import random

import sys
import pdb
import glob
import os

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

# ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
#               'l_set', 'l-spike', 'l-pass', 'l_winpoint']
ACTIVITIES = ['r-serve', 'r-block', 'r-firstpass', 'r-set', 'r-spike', 'r-winpoint', 'r-drop', 'r-volley', 'r-shot',
              'l-serve', 'l-block', 'l-firstpass', 'l-set', 'l-spike', 'l-winpoint', 'l-drop', 'l-volley', 'l-shot']

WINNERS = {'r-winpoint': 0, 'l-winpoint': 1}

NUM_ACTIVITIES = 2


ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9

def volley_winner_extract(frames, anns):
    winner_frames = []
    for item in frames:
        if anns[item[0]][item[1]]['group_activity'] == 5 or anns[item[0]][item[1]]['group_activity'] == 14:
            winner_frames.append(item)

    return winner_frames


def volley_read_annotations(path):
    """
    reading annotations for the given sequence
    """
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = values[0]
            activity = gact_to_id[values[1]]

            values = values[2:]
            num_people = len(values) // 5

            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y + h, x + w

            bboxes = np.array([_read_bbox(values[i:i + 4])
                               for i in range(0, 5 * num_people, 5)])

            fid = int(file_name.split('.')[0])
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
    return annotations


def VOLLEY_READ_ANNOTATIONS(path):
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            if 'mp4' in l:
                video_name = l.split('.mp4')[0]
            elif '-' in l:
                values = l[:-1].split(' ')
                frame_name = video_name + values[0]
                frame_start = values[1]
                frame_end = values[2]
                activity = gact_to_id[values[6]]
                actions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                bboxes = np.random.randn(12, 4)
                fid = int(frame_name)
                annotations[fid] = {
                    'file_name': frame_name,
                    'frame_start': frame_start,
                    'frame_end': frame_end,
                    'group_activity': activity,
                    'actions': actions,
                    'bboxes': bboxes,
                }
    return annotations


def volley_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = VOLLEY_READ_ANNOTATIONS(path + '/%d/group.txt' % sid)
    # pdb.set_trace()
    return data


def volley_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


def volley_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames


def volley_frames_around(frame, num_before=5, num_after=4):
    sid, src_fid = frame
    if src_fid%100000==0:
        num_before=0
        num_after=7+7
    return [(sid, src_fid, fid)
            for fid in range(src_fid - num_before, src_fid + num_after + 1)]


def load_samples_sequence(anns, tracks, images_path, frames, image_size, num_boxes=12, ):
    """
    load samples of a bath

    Returns:
        pytorch tensors
    """
    images, boxes, boxes_idx = [], [], []
    activities, actions = [], []
    for i, (sid, src_fid, fid) in enumerate(frames):
        # img=skimage.io.imread(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
        # img=skimage.transform.resize(img,(720, 1280),anti_aliasing=True)

        img = Image.open(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

        img = transforms.functional.resize(img, image_size)
        img = np.array(img)

        # H,W,3 -> 3,H,W
        img = img.transpose(2, 0, 1)
        images.append(img)

        boxes.append(tracks[(sid, src_fid)][fid])
        actions.append(anns[sid][src_fid]['actions'])
        if len(boxes[-1]) != num_boxes:
            boxes[-1] = np.vstack([boxes[-1], boxes[-1][:num_boxes - len(boxes[-1])]])
            actions[-1] = actions[-1] + actions[-1][:num_boxes - len(actions[-1])]
        boxes_idx.append(i * np.ones(num_boxes, dtype=np.int32))
        activities.append(anns[sid][src_fid]['group_activity'])

    images = np.stack(images)
    activities = np.array(activities, dtype=np.int32)
    bboxes = np.vstack(boxes).reshape([-1, num_boxes, 4])
    bboxes_idx = np.hstack(boxes_idx).reshape([-1, num_boxes])
    actions = np.hstack(actions).reshape([-1, num_boxes])

    # convert to pytorch tensor
    images = torch.from_numpy(images).float()
    bboxes = torch.from_numpy(bboxes).float()
    bboxes_idx = torch.from_numpy(bboxes_idx).int()
    actions = torch.from_numpy(actions).long()
    activities = torch.from_numpy(activities).long()

    return images, bboxes, bboxes_idx, actions, activities


class VolleyballDataset(data.Dataset):
    """
    Characterize volleyball dataset for pytorch
    """

    def __init__(self, anns, wtracks, tracks, frames, images_path, image_size, feature_size, OR=1, num_boxes=12, num_before=10,
                 num_after=10, is_training=True, is_finetune=False):
        Frames = []
        for frame in frames:
            if frame in wtracks:
                Frames.append(frame)
        self.anns = anns
        self.tracks = tracks
        self.wtracks = wtracks
        self.frames = Frames
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_boxes = num_boxes
        self.num_before = num_before
        self.num_after = num_after

        self.is_training = is_training
        self.is_finetune = is_finetune
        self.OR = OR
        

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)

    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """
        select_frames = self.volley_frames_sample(self.frames[index])
        sample = self.load_samples_sequence(select_frames)

        return sample

    def volley_frames_sample(self, frame):

        sid, src_fid = frame
        # print(src_fid)
        if src_fid%100000 == 0:
            # print(src_fid)
            num_before = 0
            num_after = 7 + 7
            # num_before = 0 # Anticipation
            # num_after = 2 # Anticipation
        elif str(src_fid+7).zfill(7) not in self.wtracks[(sid, src_fid)]:
            # print(src_fid)
            num_before=7+7
            num_after=0
            # num_before = 14 # Anticipation
            # num_after = -12 # Anticipation
        else:
            num_before=self.num_before
            num_after=self.num_after
            # num_after = self.num_after-12 # Anticipation

        if self.is_finetune:
            if self.is_training:
                fid = random.randint(src_fid - num_before, src_fid + num_after)
                return [(sid, src_fid, fid)]
            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid - num_before, src_fid + num_after + 1)]
        else:
            if self.is_training:
                # sample_frames = random.sample(range(src_fid - num_before, src_fid + num_after + 1), 3)
                clip_index = str(int((src_fid - src_fid % 100000) / 1000000)).zfill(6)
                img_list = glob.glob(os.path.join("/mnt/HDD/hanbin/volleyball/dataset/volleyball", str(sid), "frames", clip_index, "*.jpg"))
                sample_frames = random.sample(range(0, len(img_list)), 3)

                sample_frames.append(len(img_list)-3)
                sample_frames.append(len(img_list)-2)
                sample_frames.append(len(img_list)-1)

                sample_fids = []

                for fid in sample_frames:
                    sample_fids.append((sid, clip_index, fid, src_fid))

                return sample_fids
                # return [(sid, src_fid, fid)
                        # for fid in sample_frames]
            else:
                # fid=fid+int((num_after-num_before)/2)
                
                clip_index = str(int((src_fid - src_fid % 100000) / 1000000)).zfill(6)
                img_list = glob.glob(os.path.join("/mnt/HDD/hanbin/volleyball/dataset/volleyball", str(sid), "frames", clip_index, "*.jpg"))
                sample_frames = random.sample(range(0, int(len(img_list)*self.OR)), 3)
                sample_frames.append(int(len(img_list)*self.OR) - 3)
                sample_frames.append(int(len(img_list)*self.OR) -2)
                sample_frames.append(int(len(img_list)*self.OR) -1)

                # sample_frames = random.sample(range(0, len(img_list)), 3)
                # sample_frames.append(len(img_list)-3)
                # sample_frames.append(len(img_list)-2)
                # sample_frames.append(len(img_list)-1)

                sample_fids = []
                for fid in sample_frames:
                    sample_fids.append((sid, clip_index, fid, src_fid))

                return sample_fids


    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """

        OH, OW = self.feature_size

        images, boxes = [], []
        activities, actions = [], []

        for i, (sid, cid, fid, src_fid) in enumerate(select_frames):
            img = Image.open(os.path.join("/mnt/HDD/hanbin/volleyball/dataset/volleyball", str(sid), \
                "frames", cid, "%06d"%fid+".jpg"))
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)
            # H,W,3 -> 3,H,W
            img = img.transpose(2, 0, 1)
            images.append(img)

            try:
                temp_boxes = np.ones_like(self.tracks[(str(sid), cid)][str(fid).zfill(6)])
            except:
                print(sid,src_fid,str(fid))

            for j, track in enumerate(self.tracks[(str(sid), cid)][str(fid).zfill(6)]):
                x1, y1, x2, y2 = track
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                temp_boxes[j] = np.array([w1, h1, w2, h2])

            boxes.append(temp_boxes)

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes - len(boxes[-1])]])
                # actions[-1] = actions[-1] + actions[-1][:self.num_boxes - len(actions[-1])]
            
            activities.append(WINNERS[ACTIVITIES[self.anns[sid][src_fid]['group_activity']]])



        # for i, (sid, src_fid, fid) in enumerate(select_frames):
        #     key_frame = str(fid % 100000).zfill(6)
        #     clip_index = str(int((src_fid - src_fid % 100000) / 1000000)).zfill(6)
        #     try:
        #         img = Image.open(self.images_path + str(
        #         sid) + '/frames/' + clip_index + '/' + key_frame + '.jpg')  # % (sid, str(src_fid)[-6:], str(fid)[-6:]))
        #     except:
        #         print(sid,fid,key_frame)

        #     img = transforms.functional.resize(img, self.image_size)
        #     img = np.array(img)

        #     try:
        #         temp_boxes = np.ones_like(self.tracks[(str(sid), clip_index)][str(fid).zfill(6)])
        #     except:
        #         print(sid,src_fid,str(fid))

        #     for i, track in enumerate(self.tracks[(str(sid), clip_index)][str(fid).zfill(6)]):
        #         x1, y1, x2, y2 = track
        #         w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
        #         temp_boxes[i] = np.array([w1, h1, w2, h2])

        #     boxes.append(temp_boxes)

        #     # actions.append(self.anns[sid][src_fid]['actions'])

        #     if len(boxes[-1]) != self.num_boxes:
        #         boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes - len(boxes[-1])]])
        #         actions[-1] = actions[-1] + actions[-1][:self.num_boxes - len(actions[-1])]
        #     # activities.append(self.anns[sid][src_fid]['group_activity'])
        #     activities.append(WINNERS[ACTIVITIES[self.anns[sid][src_fid]['group_activity']]])


            # # H,W,3 -> 3,H,W
            # img = img.transpose(2, 0, 1)
            # images.append(img)
            # if fid<10000:
            #     fid=str(fid).zfill(7)
            # # if str(fid) not in self.tracks[(sid, src_fid)]:
            # #     fid-=7
            # try:
            #     temp_boxes = np.ones_like(self.tracks[(sid, src_fid)][str(fid)])
            # except:
            #     print(sid,src_fid,str(fid))
            # for i, track in enumerate(self.tracks[(sid, src_fid)][str(fid)]):
            #     x1, y1, x2, y2 = track
            #     w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
            #     temp_boxes[i] = np.array([w1, h1, w2, h2])

            # boxes.append(temp_boxes)

            # actions.append(self.anns[sid][src_fid]['actions'])

            # if len(boxes[-1]) != self.num_boxes:
            #     boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes - len(boxes[-1])]])
            #     actions[-1] = actions[-1] + actions[-1][:self.num_boxes - len(actions[-1])]
            # # activities.append(self.anns[sid][src_fid]['group_activity'])
            # activities.append(WINNERS[ACTIVITIES[self.anns[sid][src_fid]['group_activity']]])

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        # actions = np.hstack(actions).reshape([-1, self.num_boxes])

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        # actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()

        # return images, bboxes, actions, activities
        return images, bboxes, activities

