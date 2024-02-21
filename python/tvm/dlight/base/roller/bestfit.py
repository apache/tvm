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
"""Benifit For Dlight Schedule"""
class Block:
    def __init__(self, start, end, is_free):
        self.start = start
        self.end = end
        self.is_free = is_free

    def size(self) -> int:
        return self.end - self.start

    def merge(self, other):
        assert self.is_free == other.is_free
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)

    def __repr__(self) -> str:
        return "<Block offset={} size={}>".format(self.start, self.size())


class BestFit:
    def __init__(self, align=32):
        self.limit = 0
        self.list = []
        self.align = align

    def malloc(self, size) -> Block:
        size = (size + self.align - 1) // self.align * self.align
        found = None
        for block in self.list:
            if block.is_free and block.size() >= size:
                if not found or found.size() > block.size():
                    found = block
        if found:
            found.is_free = False
            remain = found.size() - size
            if remain != 0:
                found.end -= remain
                self.list.insert(
                    self.list.index(found) + 1, Block(found.end, found.end + remain, True)
                )
            return found
        elif len(self.list) > 0 and self.list[-1].is_free:
            add = size - self.list[-1].size()
            self.list[-1].end += add
            self.limit = self.list[-1].end
            self.list[-1].is_free = False
            return self.list[-1]
        else:
            block = Block(self.limit, self.limit + size, False)
            self.list.append(block)
            self.limit += size
            return block

    def free(self, block: Block) -> None:
        assert not block.is_free
        idx = self.list.index(block)
        self.list[idx] = Block(block.start, block.end, True)
        if idx + 1 < len(self.list) and self.list[idx + 1].is_free:
            self.list[idx].merge(self.list[idx + 1])
            self.list.pop(idx + 1)
        if idx - 1 >= 0 and self.list[idx - 1].is_free:
            self.list[idx].merge(self.list[idx - 1])
            self.list.pop(idx - 1)
