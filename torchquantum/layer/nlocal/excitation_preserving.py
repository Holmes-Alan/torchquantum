"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torchquantum as tq
from torchquantum.layer.layers import (
    LayerTemplate0,
    Op1QAllLayer,
    Op2QAllLayer,
    RandomOp1All,
)
from .two_local import TwoLocal

__all__ = [
    "ExcitationPreserving",
]

class ExcitationPreserving(TwoLocal):
    """Layer Template for a ExcitationPreserving circuit

    Args:
        arch (dict): circuit architecture in a dictionary format
        entanglement_layer (str): type of entanglement layer in a string ("linear", "reverse_linear", "circular", "full") or tq.QuantumModule format
        reps (int): number of reptitions of the rotation and entanglement layers in a integer format
        skip_final_rotation_layer (bool): whether or not to add the final rotation layer as a boolean
    """

    def __init__(
        self,
        arch: dict = None,
        entanglement_layer: str = "full",
        reps: int = 3,
        skip_final_rotation_layer: bool = False,
    ):
        # construct circuit with rotation layers of RZ and entanglement with RXX and RYY
        super().__init__(
            arch=arch,
            rotation_ops=[tq.RZ],
            entanglement_ops=[tq.RXX, tq.RYY],
            entanglement_layer=entanglement_layer,
            entanglement_layer_params={"has_params": True, "trainable": True},
            reps=reps,
            skip_final_rotation_layer=skip_final_rotation_layer,
        )
