# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import Tuple
import numpy as np
import torch

from TempCLR.utils import Array


def targets_to_array_and_indices(
    targets,
    field_key: str,
    data_key: str,
) -> Tuple[Array, Array]:
    indices = np.array([ii for ii, t in enumerate(targets) if
                        t.has_field(field_key)], dtype=np.int)
    if len(indices) > 1:
        data_lst = []
        for ii, t in enumerate(targets):
            if t.has_field(field_key):
                data = getattr(t.get_field(field_key), data_key)
                if torch.is_tensor(data):
                    data = data.detach().cpu().numpy()
                data_lst.append(data)
        data_array = np.stack(data_lst)
        return data_array, indices
    else:
        return np.array([]), indices
