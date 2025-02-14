# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""A database that stores TuningRecords in memory"""
from tvm._ffi import register_object

from .. import _ffi_api
from .database import Database


@register_object("meta_schedule.MemoryDatabase")
class MemoryDatabase(Database):
    """An in-memory database

    Parameters
    ----------
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:
          - "structural": Use StructuralEqual/Hash
    """

    def __init__(
        self,
        module_equality: str = "structural",
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.DatabaseMemoryDatabase,  # type: ignore # pylint: disable=no-member,
            module_equality,
        )
