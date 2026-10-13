import csv
import functools
import json
import os
import random
import warnings
import pickle

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    """
    total_size = len(dataset)
    
    # 1. 处理 ratio 逻辑
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1

    indices = list(range(total_size))

    # 2. 核心修复：确保所有 size 均为 int 类型
    # 使用 kwargs.get() 防止键不存在时报错，并强制转换为 int
    if kwargs.get('train_size') is not None:
        train_size = int(kwargs['train_size'])
    else:
        train_size = int(train_ratio * total_size)

    if kwargs.get('test_size') is not None:
        test_size = int(kwargs['test_size'])
    else:
        test_size = int(test_ratio * total_size)

    if kwargs.get('val_size') is not None:
        valid_size = int(kwargs['val_size'])
    else:
        valid_size = int(val_ratio * total_size)

    # 3. 创建 Sampler（此时 train_size 等确定为整数）
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])

    # 4. 构建 DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal properties.
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class GaussianDistance(object):
    """Expands the distance by Gaussian basis."""
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """Base class for intializing the vector representation for atoms."""
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """The CIFData dataset wrapper."""
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 disable_save_torch=False, random_seed=42):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.disable_save_torch = disable_save_torch
        
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), f'File not found: {id_prop_file}'
        
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            # 移除可能的 BOM 字符，并跳过表头
            rows = [[x.strip().replace('\ufeff', '') for x in row] 
                    for row in reader if row]
            # 检测并跳过表头（如果第一行包含 'EA' 或 'IP' 等非数字）
            if rows and not self._is_numeric(rows[0][1]):
                print(f"检测到CSV表头，已跳过: {rows[0]}")
                self.id_prop_data = rows[1:]
            else:
                self.id_prop_data = rows
        
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        
        self.torch_data_path = os.path.join(self.root_dir, 'cifdata')
        if not os.path.exists(self.torch_data_path):
            os.makedirs(self.torch_data_path)

    def __len__(self):
        return len(self.id_prop_data)
    
    def _is_numeric(self, value):
        """检查字符串是否为数字"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        # 进一步清理 ID
        cif_id = cif_id.replace('\ufeff', '').replace('ï»¿', '')
        target = torch.Tensor([float(target)])

        pkl_path = os.path.join(self.torch_data_path, cif_id + '.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                atom_fea, nbr_fea, nbr_fea_idx = pickle.load(f)
        else:
            cif_path = os.path.join(self.root_dir, cif_id + '.cif')
            crystal = Structure.from_file(cif_path)
            
            # 原子特征
            atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                                  for i in range(len(crystal))])
            atom_fea = torch.Tensor(atom_fea)
            
            # 邻居搜索
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    # 邻居不足补 0
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
            
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)
            
            nbr_fea = torch.Tensor(nbr_fea)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
            
            if not self.disable_save_torch:
                with open(pkl_path, 'wb') as f:
                    pickle.dump((atom_fea, nbr_fea, nbr_fea_idx), f)

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


# ==========================================================
# MultiTargetCIFData - 支持多目标值（EA, IP等）
# ==========================================================
class MultiTargetCIFData(CIFData):
    """
    扩展CIFData以支持多个目标值（如EA和IP）
    id_prop.csv格式（无表头）: material_id, target1, target2, ...
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 disable_save_torch=False, random_seed=42):
        super().__init__(root_dir, max_num_nbr, radius, dmin, step,
                        disable_save_torch, random_seed)
    
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        # 解包所有列：第一列是ID，其余列是目标值
        row = self.id_prop_data[idx]
        cif_id = row[0]
        targets = [float(x) for x in row[1:]]  # 所有目标值
        
        # 清理ID
        cif_id = cif_id.replace('\ufeff', '').replace('ï»¿', '')
        target = torch.Tensor(targets)  # 多维张量

        pkl_path = os.path.join(self.torch_data_path, cif_id + '.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                atom_fea, nbr_fea, nbr_fea_idx = pickle.load(f)
        else:
            cif_path = os.path.join(self.root_dir, cif_id + '.cif')
            crystal = Structure.from_file(cif_path)
            
            # 原子特征
            atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                                  for i in range(len(crystal))])
            atom_fea = torch.Tensor(atom_fea)
            
            # 邻居搜索
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
            
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)
            
            nbr_fea = torch.Tensor(nbr_fea)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
            
            if not self.disable_save_torch:
                with open(pkl_path, 'wb') as f:
                    pickle.dump((atom_fea, nbr_fea, nbr_fea_idx), f)

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id