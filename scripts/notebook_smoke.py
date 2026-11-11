"""Exercise the actual notebook model definitions on CPU without training a corpus.

This is a shape/gradient/causal-mask smoke check, not a quality benchmark.
"""
import ast
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]

def check_notebook(name, class_name):
    torch.manual_seed(42)
    nb = json.loads((ROOT / name).read_text())
    namespace = dict(torch=torch, nn=nn, F=F, device='cpu', block_size=8,
                     batch_size=2, n_embd=16, n_head=2, n_layer=1, dropout=0.0)
    definitions = []
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        module = ast.parse(''.join(cell['source']))
        definitions.extend(node for node in module.body
                           if isinstance(node, (ast.ClassDef, ast.FunctionDef)))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), name, 'exec'), namespace)
    model = namespace[class_name](vocab_size=11)
    x = torch.randint(0, 11, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (16, 11)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    model.eval()
    with torch.no_grad():
        generated = model.generate(x[:, :1], max_new_tokens=3)
        assert generated.shape == (2, 4)
        # Changing a future token must not change the first position's logits.
        other = x.clone()
        other[:, -1] = (other[:, -1] + 1) % 11
        first, _ = model(x)
        second, _ = model(other)
        assert torch.allclose(first[:, 0], second[:, 0], atol=1e-6)
    namespace['train_data'] = torch.arange(40) % 11
    namespace['val_data'] = torch.arange(20) % 11
    assert namespace['get_batch']('train')[0].shape == (2, 8)
    try:
        namespace['get_batch']('typo')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown split silently accepted')
    print(f'{name}: forward, backward, generation, causal and batch checks passed')

if __name__ == '__main__':
    check_notebook('bigram.ipynb', 'BigramLanguageModel')
    check_notebook('gpt-v1 (3).ipynb', 'GPTLanguageModel')
