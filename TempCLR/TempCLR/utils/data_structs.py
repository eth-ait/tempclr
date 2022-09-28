# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from dataclasses import make_dataclass, fields, field
from loguru import logger


class Struct(object):
    def __new__(cls, **kwargs):
        class_fields = [
            [key, type(val), field(default=val)]
            for key, val in kwargs.items()
        ]

        object_type = make_dataclass(
            'Struct',
            class_fields,
            namespace={
                'keys': lambda self: [f.name for f in fields(self)],
            })
        return object_type()
