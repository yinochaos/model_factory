#!/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2020 yinochaos <pspcxl@163.com>. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" model interface
"""
import pydoc
from typing import Dict, Any

def load_data_object(data: Dict, **kwargs: Dict) -> Any:
    """
    Load Object From Dict
    Args:
        data:
        **kwargs:
    Returns:
    """
    module_name = f"{data['__module__']}.{data['__class_name__']}"
    obj: Any = pydoc.locate(module_name)(**data['config'], **kwargs)  # type: ignore
    if hasattr(obj, '_override_load_model'):
        obj._override_load_model(data)

    return obj