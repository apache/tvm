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
"""Rasteration Plan For L2 Cache Locality"""

from typing import List


class Rasterization:
    def __init__(self) -> None:
        pass

    def get_code(self) -> List[str]:
        raise NotImplementedError()


class NoRasterization(Rasterization):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "<NoRasterization>"

    def get_code(self) -> List[str]:
        return []


class Rasterization2DRow(Rasterization):
    """
    Rasterization by Row, each Row line width is panel_width
         _________
         _________|
        |_________
        __________|
    """

    def __init__(self, panel_width=4) -> None:
        super().__init__()
        self.panel_width_ = panel_width

    def __repr__(self) -> str:
        return f"<Rasterization2DRow({self.panel_width_})>"

    def get_code(self) -> List[str]:
        raise NotImplementedError()


class Rasterization2DColumn(Rasterization):
    """
    Rasterization by Column, each column line width is panel_width
            _
         | | | |
         | | | |
         |_| |_|
    """

    def __init__(self, panel_width=4) -> None:
        super().__init__()
        self.panel_width_ = panel_width

    def __repr__(self) -> str:
        return f"<Rasterization2DColumn({self.panel_width_})>"

    def get_device_function(self) -> str:
        return """
__device__ dim3 rasterization2DColumn(const int panel_width) {
    const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
    const auto totalPanel = (gridDim.x * gridDim.y +panel_width * gridDim.x - 1) / (panel_width * gridDim.x);
    const auto totalBlock = gridDim.x * gridDim.y;
    const auto panelIdx = baseBlockIdx / (panel_width *gridDim.x);
    const auto strideLd = panelIdx + 1 < totalPanel ?panel_width : (totalBlock - panelIdx * (panel_width *gridDim.x)) / gridDim.x;
    const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * panel_width * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * panel_width *gridDim.x) / strideLd;
    const auto by = (baseBlockIdx - panelIdx * panel_width *gridDim.x) % strideLd + panelIdx * panel_width;
    const auto bz = blockIdx.z;
    
    dim3 blockIdx(bx, by, bz);
    return blockIdx;
}
    """

    def get_code(self) -> List[str]:
        return [
            self.get_device_function(),
            "const dim3 blockIdx(rasterization2DColumn({});".format(self.panel_width_),
        ]
