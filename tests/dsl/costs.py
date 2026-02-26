from faran import types

from numtypes import shape_of

import jax.numpy as jnp
import numpy as np


type NumPyControlInputBatch = types.numpy.ControlInputBatch
type NumPyStateBatch = types.numpy.StateBatch
type NumPyCosts = types.numpy.Costs
type NumPyCostFunction = types.numpy.CostFunction

type JaxControlInputBatch = types.jax.ControlInputBatch
type JaxStateBatch = types.jax.StateBatch
type JaxCosts = types.jax.Costs
type JaxCostFunction = types.jax.CostFunction


class numpy:
    @staticmethod
    def energy() -> NumPyCostFunction:
        """Cost function that penalizes control energy: cost = sum(u^2)."""

        def energy_cost(
            inputs: NumPyControlInputBatch, states: NumPyStateBatch
        ) -> NumPyCosts:
            T, M = inputs.horizon, inputs.rollout_count

            costs = np.sum(np.asarray(inputs) ** 2, axis=1)

            assert shape_of(costs, matches=(T, M), name="energy costs")

            return types.numpy.simple.costs(costs)

        return energy_cost

    @staticmethod
    def quadratic_distance_to_origin() -> NumPyCostFunction:
        """Cost function that penalizes distance from origin: cost = ||x||^2."""

        def quadratic_cost(
            inputs: NumPyControlInputBatch, states: NumPyStateBatch
        ) -> NumPyCosts:
            states_array = np.asarray(states)
            return types.numpy.simple.costs(np.sum(states_array**2, axis=1))

        return quadratic_cost


class jax:
    @staticmethod
    def energy() -> JaxCostFunction:
        """Cost function that penalizes control energy: cost = sum(u^2)."""

        def energy_cost(
            inputs: JaxControlInputBatch, states: JaxStateBatch
        ) -> JaxCosts:
            return types.jax.simple.costs(jnp.sum(inputs.array**2, axis=1))

        return energy_cost

    @staticmethod
    def quadratic_distance_to_origin() -> JaxCostFunction:
        """Cost function that penalizes distance from origin: cost = ||x||^2."""

        def quadratic_cost(
            inputs: JaxControlInputBatch, states: JaxStateBatch
        ) -> JaxCosts:
            return types.jax.simple.costs(jnp.sum(states.array**2, axis=1))

        return quadratic_cost
