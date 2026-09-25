#!/usr/bin/env python
# coding: utf-8

# # Dataset

# In[7]:


import os
import copy
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import csv
import math
import SimpleITK as sitk
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
from util.disk import getCache
from functools import lru_cache
from glob import glob
from collections import namedtuple


# In[8]:


data_dir = "./data/"

CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple", "isNodule_bool, diameter_mm, series_uid, center_xyz"
)


@lru_cache(1)
def getCandidateList(requireOnDisk_bool=True):
    mhd_list = glob(data_dir + "subset*/*.mhd")
    presentOnDist = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open(data_dir + "annotations.csv", "r") as f:
        for row in list(csv.reader(f))[1:]:
            anno_seriesuid = row[0]
            anno_centerxyz = [float(x) for x in row[1:4]]
            anno_diameter_mm = float(row[4])

            diameter_dict.setdefault(anno_seriesuid, []).append(
                (anno_centerxyz, anno_diameter_mm)
            )

    candidate_list = []
    RATIO = (1 + math.sqrt(3)) / 8
    with open(data_dir + "candidates.csv", "r") as f:
        for row in list(csv.reader(f))[1:]:
            cand_seriesuid = row[0]

            if cand_seriesuid not in presentOnDist and requireOnDisk_bool:
                continue

            cand_centerxyz = [float(i) for i in row[1:4]]
            isNodule_bool = bool(int(row[4]))
            cand_diameter_mm = 0.0

            for anno_centerxyz, anno_diameter_mm in diameter_dict.get(
                cand_seriesuid, []
            ):
                threshold = RATIO * anno_diameter_mm
                delta_xyz = math.sqrt(
                    sum((c - a) ** 2 for c, a in zip(cand_centerxyz, anno_centerxyz))
                )

                if delta_xyz <= threshold:
                    cand_diameter_mm = anno_diameter_mm

            candidate_list.append(
                CandidateInfoTuple(
                    isNodule_bool, cand_diameter_mm, cand_seriesuid, cand_centerxyz
                )
            )

    candidate_list.sort(reverse=True)
    return candidate_list


# In[9]:


IrcTuple = namedtuple("IrcTuple", ["index", "row", "col"])
XyzTuple = namedtuple("XyzTuple", ["x", "y", "z"])


def irc2xyz(coordinate_irc, origin_xyz, vxSize, direction_a):
    cri_a = np.array(coordinate_irc)[::-1]
    origin_xyz = np.array(origin_xyz)
    vxSize = np.array(vxSize)

    return XyzTuple(direction_a @ (cri_a * vxSize) + origin_xyz)


def xyz2irc(coordinate_xyz, origin_xyz, vxSize, direction_a):
    coord_a = np.array(coordinate_xyz)
    origin_xyz = np.array(origin_xyz)
    vxSize = np.array(vxSize)

    coord_a = np.round(
        ((coordinate_xyz - origin_xyz) @ np.linalg.inv(direction_a)) / vxSize
    )
    return IrcTuple(int(coord_a[2]), int(coord_a[1]), int(coord_a[0]))


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob(data_dir + "subset*/{}.mhd".format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize, self.direction_a)
        slice_list = []

        for axis, center in enumerate(center_irc):
            w = width_irc[axis]
            slice_list.append(
                slice(int(round(center - w / 2)), int(round(center + w / 2)))
            )

        ct_sliced = self.hu_a[tuple(slice_list)]
        return ct_sliced, center_irc


# In[10]:


raw_cache = getCache("part2ch10_raw")


@lru_cache(1, typed=True)
def getct(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getct(series_uid)
    ct_sliced, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_sliced, center_irc


# In[5]:


class LunaDataset(Dataset):
    def __init__(
        self,
        isValidation,
        val_stride,
        series_uid,
    ):
        self.candidate_list = copy.copy(getCandidateList())

        if series_uid:
            self.candidate_list = [
                candidate
                for candidate in self.candidate_list
                if candidate.series_uid == series_uid
            ]

        if isValidation:
            self.candidate_list = self.candidate_list[::val_stride]
        elif val_stride > 0:
            del self.candidate_list[::val_stride]

    def __len__(self):
        return len(self.candidate_list)

    def __getitem__(self, ndx):
        candidate_tup = self.candidate_list[ndx]
        isNodule_bool, diameter_mm, series_uid, center_xyz = candidate_tup
        width_irc = (32, 48, 48)

        ct_sliced, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)

        candidate_t = (torch.from_numpy(ct_sliced)).to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([not isNodule_bool, isNodule_bool], dtype=torch.long)

        return (
            candidate_t,
            pos_t,
            series_uid,
            torch.tensor(center_irc),
        )
