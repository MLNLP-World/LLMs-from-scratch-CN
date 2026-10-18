# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

import json
from pathlib import Path

import pytest
import torch

from gpt import GPTModel, LayerNorm, TransformerBlock, main

expected = """
==================================================
                      IN
==================================================

Input text: Hello, I am
Encoded input text: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])


==================================================
                      OUT
==================================================

Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
         49706, 43231, 47062, 34657]])
Output length: 14
Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous
"""


def test_main(capsys):
    main()
    captured = capsys.readouterr()

    # Normalize line endings and strip trailing whitespace from each line
    normalized_expected = '\n'.join(line.rstrip() for line in expected.splitlines())
    normalized_output = '\n'.join(line.rstrip() for line in captured.out.splitlines())

    # Compare normalized strings
    assert normalized_output == normalized_expected


@pytest.mark.parametrize("n_layers", [1, 3, 12])
def test_notebook_model_matches_standalone(n_layers):
    notebook = json.loads(Path(__file__).with_name("ch04.ipynb").read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "".join(cell["source"]).startswith("class GPTModel(")
    )
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "LayerNorm": LayerNorm,
        "TransformerBlock": TransformerBlock,
    }
    exec(source, namespace)
    cfg = {
        "vocab_size": 16,
        "context_length": 4,
        "emb_dim": 8,
        "n_heads": 2,
        "n_layers": n_layers,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    torch.manual_seed(123)
    notebook_model = namespace["GPTModel"](cfg)
    torch.manual_seed(123)
    standalone_model = GPTModel(cfg)

    assert len(notebook_model.trf_blocks) == n_layers
    tokens = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(notebook_model(tokens), standalone_model(tokens))
