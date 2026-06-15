"""
antpilot/rough_env.py
RoughCmdAnt — separate rough-terrain version of CmdAnt.

This environment is NOT used for training.
It is only meant for evaluating already trained models on uneven terrain.

It uses a custom Ant XML file with a MuJoCo heightfield.

Expected XML path:
    antpilot/assets/ant.xml
"""

import os
import numpy as np
import gymnasium as gym

from .env import CmdAnt
from .config import DEFAULT_TERRAIN_ROUGHNESS


class RoughCmdAnt(CmdAnt):
    """
    Rough-terrain variant of CmdAnt.

    This class reuses all command observation logic and reward functions from CmdAnt,
    but creates the underlying Ant-v5 environment from a custom XML file that contains
    a heightfield named "terrain".

    The terrain is regenerated on every reset.
    """

    def __init__(
        self,
        command: np.ndarray = None,
        stage_probs: dict[str, float] = None,
        render_mode: str = None,
        terrain_roughness: float = None,
        terrain_seed: int = None,
    ):
        if terrain_roughness is None:
            terrain_roughness = DEFAULT_TERRAIN_ROUGHNESS

        self.terrain_roughness = float(terrain_roughness)
        self.terrain_seed = terrain_seed

        xml_file = os.path.join(
            os.path.dirname(__file__),
            "assets",
            "ant.xml",
        )

        base_env = gym.make(
            "Ant-v5",
            xml_file=xml_file,
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,
        )

        super().__init__(
            command=command,
            stage_probs=stage_probs,
            render_mode=render_mode,
            base_env=base_env,
        )

    # ------------------------------------------------------------------
    # Heightfield helpers
    # ------------------------------------------------------------------

    def _hfield_slice(self):
        model = self.env.unwrapped.model

        try:
            hfield_id = model.hfield("terrain").id
        except Exception:
            return None

        start = model.hfield_adr[hfield_id]
        nrow = model.hfield_nrow[hfield_id]
        ncol = model.hfield_ncol[hfield_id]
        size = nrow * ncol

        return model, start, size, nrow, ncol

    def _clear_terrain(self):
        data = self._hfield_slice()

        if data is None:
            return

        model, start, size, _nrow, _ncol = data
        model.hfield_data[start:start + size] = 0.0

    def _smooth_heightfield(
        self,
        terrain: np.ndarray,
        passes: int = 4,
    ) -> np.ndarray:
        for _ in range(passes):
            terrain = (
                terrain
                + np.roll(terrain, 1, axis=0)
                + np.roll(terrain, -1, axis=0)
                + np.roll(terrain, 1, axis=1)
                + np.roll(terrain, -1, axis=1)
            ) / 5.0

        return terrain

    def _generate_terrain(self):
        data = self._hfield_slice()

        if data is None:
            return

        model, start, size, nrow, ncol = data

        if self.terrain_seed is not None:
            rng = np.random.default_rng(self.terrain_seed)
            terrain = rng.normal(
                loc=0.0,
                scale=1.0,
                size=(nrow, ncol),
            )
        else:
            terrain = self.env.unwrapped.np_random.normal(
                loc=0.0,
                scale=1.0,
                size=(nrow, ncol),
            )

        terrain = self._smooth_heightfield(terrain, passes=4)

        # Normalize to 0..1.
        terrain -= terrain.min()

        max_h = terrain.max()
        if max_h > 1e-8:
            terrain /= max_h

        # terrain_roughness controls how much of the XML height is used.
        terrain *= self.terrain_roughness

        model.hfield_data[start:start + size] = terrain.ravel()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kw):
        raw, info = self.env.reset(**kw)

        self._generate_terrain()

        return self._obs(raw, info), info