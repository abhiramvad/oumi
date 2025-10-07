# Copyright 2025 - Oumi
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

from typing_extensions import override

from oumi.core.datasets import BaseDpoDataset
from oumi.core.registry import register_dataset


@register_dataset("HumanLLMs/Human-Like-DPO-Dataset")
class HumanLikeDpoDataset(BaseDpoDataset):
    """Human-Like DPO Dataset for preference tuning.

    A dataset designed for Direct Preference Optimization (DPO) training
    containing human preference pairs for training language models.

    The dataset is available on HuggingFace Hub at:
    https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset

    Expected format:
        {
            "prompt": "How is the weather in Tokyo?",
            "chosen": "It's sunny and warm.",
            "rejected": "It's rainy and cold."
        }
    """

    default_dataset = "HumanLLMs/Human-Like-DPO-Dataset"

    @override
    def transform_preference(self, example: dict) -> dict:
        """Transform the raw dataset example to Oumi's DPO format.

        Args:
            example: A dictionary containing 'prompt', 'chosen', and 'rejected' keys.

        Returns:
            A dictionary with properly formatted 'prompt', 'chosen', and 'rejected'
            in message list format.
        """
        # The HumanLLMs dataset already has the correct keys (prompt, chosen, rejected)
        # so we can use the parent class's logic directly
        return super().transform_preference(example)
