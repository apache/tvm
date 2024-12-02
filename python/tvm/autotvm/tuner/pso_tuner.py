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
"""Particle Swarm Optimization Tuner"""
import logging

import numpy as np

from .tuner import Tuner
from ..utils import format_si_prefix

logger = logging.getLogger("autotvm")


def point2knob(p, dims):
    """
    Convert a point form (single integer) to a knob form (vector).

    Parameters
    ----------
    p (int): The point form to be converted.
    dims (list or np.array): The dimensions of the space in which the point exists.

    Returns
    ----------
    list: The knob form of the point.
    """
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim
    return knob


def knob2point(knob, dims) -> int:
    """
    Convert a knob form (vector) to a point form (single integer)

    Parameters
    ----------
    knob (list or np.array): The knob vector to be converted.
    dims (list or np.array): The dimensions of the space in which the knob vector exists.

    Returns
    ----------
    int: The point form of the knob vector.
    """
    p = 0
    for j, k in enumerate(knob):
        p += int(np.prod(dims[:j])) * k
    return p


class Particle:
    """
    A class used to represent a Particle in Particle Swarm Optimization (PSO).

    Parameters
    ----------
    position_range (np.array): The range of the position values.
    velocity_range (np.array): The range of the velocity values.
    dimension (int): The number of dimensions.
    w (float): The inertia weight.
    c1 (float): The cognitive weight.
    c2 (float): The social weight.
    """

    def __init__(
        self,
        position_range: np.array,
        velocity_range: np.array,
        dimension: int,
        w: float,
        c1: float,
        c2: float,
    ):  # pylint: disable=invalid-name,line-too-long
        # Vit+1 = wVit + c1r1(Pbestit - Pit) + c2r2(Pbestglobal - Pit)
        self.__w = w
        self.__c1 = c1
        self.__c2 = c2
        self.__position_range = position_range[:]
        self.__velocity_range = velocity_range[:]
        self.__dimension = dimension
        self.__position = self.__random_position(position_range)
        self.__velocity = self.__random_velocity(velocity_range)
        self.__best_position = np.random.randint((dimension,))
        self.__fitness = 0.0

    @staticmethod
    def __random_position(position_range):  # pylint: disable=missing-function-docstring
        return np.array([np.random.randint(low, high) for low, high in position_range])

    @staticmethod
    def __random_velocity(velocity_range):  # pylint: disable=missing-function-docstring
        return np.array([np.random.randint(low, high) for low, high in velocity_range])

    def __randint(self, low, high):
        if low >= high:
            return low
        return np.random.randint(low, high)

    def __fun_fitness(self, x) -> float:
        tile_f = x[0]
        tile_y = x[1]
        tile_x = x[2]
        tile_rc = x[3]
        tile_ry = x[4]
        tile_rx = x[5]
        auto_unroll_max_step = x[6]
        unroll_explicit = x[7]
        # """
        # tile_f 220
        # tile_y 4
        # tile_x 4
        # tile_rc 10
        # tile_ry 2
        # tile_rx 2
        # auto_unroll_max_step 3
        # unroll_explicit 2
        # """
        # Best 128, 4, 4, 8, 2, 1, 3, 1
        # Seco 64, 2, 4, 8, 2, 1, 3, 1
        fit = (
            (128 - tile_f) ** 2
            + (4 - tile_y) ** 2
            + (4 - tile_x) ** 2
            + (8 - tile_rc) ** 2
            + (2 - tile_ry) ** 2
            + (1 - tile_rx) ** 2
            + (3 - auto_unroll_max_step) ** 2
            + (1 - unroll_explicit) ** 2
        )  # pylint: disable=line-too-long

        return fit

    def get_best_position(self) -> int:  # pylint: disable=missing-function-docstring
        return self.__best_position

    def set_position(self, pos):  # pylint: disable=missing-function-docstring
        for index in range(self.__dimension):
            if pos[index] < self.__position_range[index][0]:
                self.__position[index] = self.__position_range[index][0]
            elif pos[index] > self.__position_range[index][1]:
                self.__position[index] = self.__position_range[index][1]
            else:
                self.__position[index] = pos[index]

    def set_velocity(self, vel):  # pylint: disable=missing-function-docstring
        for index in range(self.__dimension):
            if vel[index] < self.__velocity_range[index][0]:
                self.__velocity[index] = self.__velocity_range[index][0]
            elif vel[index] > self.__velocity_range[index][1]:
                self.__velocity[index] = self.__velocity_range[index][1]
            else:
                self.__velocity[index] = vel[index]

    def get_position(self):  # pylint: disable=missing-function-docstring
        return self.__position[:]

    def get_velocity(self):  # pylint: disable=missing-function-docstring
        return self.__velocity[:]

    def random_position(self):  # pylint: disable=missing-function-docstring
        self.__position = np.array(
            [self.__randint(low, high) for low, high in self.__position_range]
        )  # pylint: disable=line-too-long
        self.__best_position = np.random.randint((self.__dimension,))
        self.__fitness = 0.0

    def get_fitness(self) -> float:  # pylint: disable=missing-function-docstring
        return self.__fitness

    def update_velocity(self, global_best_position):  # pylint: disable=missing-function-docstring
        v = (
            self.__w * self.__velocity
            + self.__c1 * np.random.rand() * (self.__best_position - self.__position)
            + self.__c2 * np.random.rand() * (global_best_position - self.__position)
        )  # pylint: disable=line-too-long
        v = np.floor(v)
        self.set_velocity(v)

    # fit is flops
    def update_fitness(self, fit_flops=None) -> float:  # pylint: disable=missing-function-docstring
        if fit_flops is None:
            fit = self.__fun_fitness(self.__position)
        else:
            fit = fit_flops

        if fit > self.__fitness:
            self.__fitness = fit
            self.__best_position = self.__position
        return self.__fitness

    def update_position(self):  # pylint: disable=missing-function-docstring
        pos = self.__position + self.__velocity
        self.set_position(pos)


class PSOTuner(Tuner):
    """
    A class used to represent a Particle Swarm Optimization (PSO) tuner.

    Parameters
    ----------
    task (Task): The tuning task.
    n_particles (int): The number of particles in the swarm.
    n_iter_max (int): The maximum number of iterations.
    w (float): The inertia weight.
    c1 (float): The cognitive weight.
    c2 (float): The social weight.
    """

    def __init__(
        self,
        task,
        n_particles: int = 50,
        n_iter_max: int = 20,
        w: float = 0.5,
        c1: float = 1.4,
        c2: float = 1.4,
    ):  # pylint: disable=line-too-long
        super(PSOTuner, self).__init__(task)

        self.kv = list(self.task.config_space.space_map.items())  # (k, v)
        self.position_range = []
        self.velocity_range = []
        self.n_dimensions = len(self.kv)  # Get the number of parameters in ConfigSpace
        self.dims = []  # Used to record the number of parameters (dimension length) for each knob
        self.xs = []  # pylint: disable=invalid-name
        self.ys = []  # pylint: disable=invalid-name

        self.flops_max = 0

        logger.debug("PSOTuner kv: %s", self.kv)
        for _, value in self.kv:
            entities = value.entities  # type() == list
            self.position_range.append((0, len(entities) - 1))
            self.velocity_range.append((-100, 100))
            self.dims.append(len(value))

        self.position_range = np.array(
            self.position_range
        )  # [ [min,max], [min,max], [min,max] ... ] # pylint: disable=line-too-long

        max_possible_index = knob2point(self.position_range[:, 1], self.dims)

        self.__list_particle = [
            Particle(self.position_range, self.velocity_range, self.n_dimensions, w, c1, c2)
            for _ in range(n_particles)
        ]  # pylint: disable=line-too-long
        self.n_particles = n_particles
        self.n_iter_max = n_iter_max
        self.n_iter_cur = 0

        self.best_fitness = 0.0
        self.best_position = np.zeros(self.n_dimensions)
        self.visited = set()
        self.unvisited = set(range(max_possible_index))

        logger.debug("PSOTuner position_range: %s", self.position_range)
        logger.debug("PSOTuner max possible index: %s", max_possible_index)
        logger.debug("PSOTuner velocity_range: %s", self.velocity_range)
        logger.debug("PSOTuner dimensions (dims): %s", self.dims)
        logger.debug("PSOTuner particles: %d", len(self.__list_particle))
        logger.debug("PSOTuner n_iter_max: %d", self.n_iter_max)
        logger.debug("PSOTuner w: %.2f, c1: %.2f, c2: %.2f", w, c1, c2)

    # batch_size should equals to n_particle
    def next_batch(self, batch_size):
        ret = []
        for i in range(self.n_particles):
            particle = self.__list_particle[i]
            index = knob2point(particle.get_position(), self.dims)

            while index in self.visited:
                new_index = np.random.choice(list(self.unvisited))
                particle.set_position(point2knob(new_index, self.dims))
                index = knob2point(particle.get_position(), self.dims)
                assert new_index == index

            if self.task.config_space.is_index_valid(index):
                ret.append(self.task.config_space.get(index))
                self.visited.add(index)
                self.unvisited.remove(index)
        self.n_iter_cur = self.n_iter_cur + 1
        logger.debug("\n\nPSOTuner Iteration: %4d, BatchSize: %4d", self.n_iter_cur, len(ret))
        return ret

    def update(self, inputs, results):
        si_prefix = "G"
        # Validate si_prefix argument
        format_si_prefix(0, si_prefix)

        for inp, res in zip(inputs, results):
            index = inp.config.index
            if res.error_no == 0:
                flops = inp.task.flop / np.mean(res.costs)
            else:
                flops = 0.0
            self.xs.append(index)
            self.ys.append(flops)
            self.flops_max = max(self.flops_max, flops)

            for i in range(self.n_particles):
                particle = self.__list_particle[i]
                p_index = knob2point(self.__list_particle[i].get_position(), self.dims)

                # Find the index of the particle that is currently (input, result)
                if p_index == index:
                    fit = particle.update_fitness(flops)
                    logger.info(
                        "Iteration: %d, Particle: %d, Current/Best: %.2f/%.2f %sFLOPS, config: %s",
                        self.n_iter_cur,
                        i + 1,
                        format_si_prefix(flops, si_prefix),
                        format_si_prefix(fit, si_prefix),
                        si_prefix,
                        particle.get_position(),
                    )  # pylint: disable=line-too-long

                    if fit > self.best_fitness:
                        self.best_fitness = fit
                        self.best_position = np.copy(particle.get_position())
                        logger.debug(
                            "Iteration: %d Particle: %d, New Best: %.4f %sFLOPS, config: %s",
                            self.n_iter_cur,
                            i + 1,
                            format_si_prefix(self.best_fitness, si_prefix),
                            si_prefix,
                            self.best_position,
                        )  # pylint: disable=line-too-long
                    particle.update_velocity(self.best_position)
                    particle.update_position()
                    break

    def has_next(self):
        logger.debug("has next: %d", self.n_iter_cur < self.n_iter_max)
        return self.n_iter_cur < self.n_iter_max

    def load_history(
        self, data_set, min_seed_records=500
    ):  # pylint: disable=missing-function-docstring
        raise NotImplementedError()
