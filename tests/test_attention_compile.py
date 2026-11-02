"""Attention backend selection and CUDA backward regressions."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from dlshogi.network.policy_value_network import _default_backend, compile_model, set_compile_backend
from dlshogi.network.policy_value_network_exp__by import AttentionBlock

CUDA_INPUT = SimpleNamespace(is_cuda=True)
FLEX = 'dlshogi.network.policy_value_network_exp__by._compiled_flex_attention'


@pytest.fixture
def flex_available():
    with patch(FLEX, object()):
        yield


def impl(block, *, compiling=True, onnx=False, x=CUDA_INPUT):
    with patch('torch.compiler.is_compiling', return_value=compiling), \
         patch('torch.onnx.is_in_onnx_export', return_value=onnx):
        return block._attention_impl(x)


@pytest.mark.parametrize('name, expected', [('nt', 'aot_eager'), ('posix', None)])
def test_default_backend(monkeypatch, name, expected):
    with monkeypatch.context() as context:
        context.setattr('dlshogi.network.policy_value_network.os.name', name)
        actual = _default_backend()
    assert actual == expected


@pytest.mark.parametrize('default_backend, backend, options, expected', [
    (None, None, {}, {}),
    ('aot_eager', None, {}, {'backend': 'aot_eager'}),
    ('aot_eager', 'inductor', {}, {'backend': 'inductor'}),
    (None, 'aot_eager', {'mode': 'default', 'fullgraph': True, 'dynamic': True},
     {'backend': 'aot_eager', 'mode': 'default', 'fullgraph': True, 'dynamic': True}),
])
def test_compile_model_options_and_notification(default_backend, backend, options, expected):
    block = AttentionBlock(8, 2)
    logger = Mock()

    def capture(model, **kwargs):
        assert model is block
        assert kwargs == expected
        assert block._compile_backend == expected.get('backend')
        logger.info.assert_called_once()
        return compiled

    compiled = object()
    with patch('dlshogi.network.policy_value_network._default_backend', return_value=default_backend), \
         patch('torch.compile', side_effect=capture):
        assert compile_model(block, backend=backend, logger=logger, **options) is compiled


def test_compile_model_requires_torch_compile(monkeypatch):
    monkeypatch.delattr(torch, "compile")
    with pytest.raises(RuntimeError, match='torch.compile is not available'):
        compile_model(torch.nn.Identity())


def test_auto_backend_respects_compile_backend_and_can_switch(flex_available):
    block = AttentionBlock(8, 2)
    model = torch.nn.Sequential(block)
    set_compile_backend(model, 'aot_eager')
    assert impl(block) == 'sdpa'
    assert block.attn_backend == 'auto'
    set_compile_backend(model, 'inductor')
    assert impl(block) == 'flex'
    set_compile_backend(model, None)
    assert impl(block) == 'flex'


@pytest.mark.parametrize('backend, expected', [
    (None, 'flex'), ('inductor', 'flex'), ('aot_eager', 'sdpa'),
    ('cudagraphs', 'sdpa'), ('eager', 'sdpa'),
    (lambda graph, inputs: graph.forward, 'sdpa'),
], ids=['default', 'inductor', 'aot_eager', 'cudagraphs', 'eager', 'callable'])
def test_auto_allows_flex_only_under_inductor(backend, expected, flex_available):
    block = AttentionBlock(8, 2)
    set_compile_backend(block, backend)
    assert impl(block) == expected


@pytest.mark.parametrize('attention', ['sdpa', 'math'])
def test_explicit_attention_backend_is_preserved(attention):
    block = AttentionBlock(8, 2, attn_backend=attention)
    set_compile_backend(block, 'aot_eager')
    assert impl(block) == attention


@pytest.mark.parametrize('compiling', [False, True])
@pytest.mark.parametrize('attention', ['auto', 'sdpa', 'flex', 'math'])
def test_onnx_always_uses_math(compiling, attention, flex_available):
    block = AttentionBlock(8, 2, attn_backend=attention)
    assert impl(block, compiling=compiling, onnx=True) == 'math'


@pytest.mark.parametrize('attention', ['auto', 'sdpa', 'flex', 'math'])
@pytest.mark.parametrize('case', ['cpu', 'fused'])
def test_inference_overrides_explicit_backend(attention, case, flex_available):
    block = AttentionBlock(8, 2, attn_backend=attention)
    block._fused = case == 'fused'
    x = SimpleNamespace(is_cuda=case != 'cpu')
    assert impl(block, x=x) == 'math'


def test_eager_auto_and_missing_flex_use_sdpa():
    block = AttentionBlock(8, 2)
    assert impl(block, compiling=False) == 'sdpa'
    with patch(FLEX, None):
        assert impl(block) == 'sdpa'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('backend, fullgraph', [
    ('aot_eager', False), ('cudagraphs', False), ('inductor', True),
])
def test_auto_cuda_backward(backend, fullgraph):
    block = AttentionBlock(192, 4).cuda().train()
    compiled = compile_model(block, backend=backend, fullgraph=fullgraph)
    assert block._compile_backend == backend
    x = torch.randn(2, 192, 9, 9, device='cuda', requires_grad=True)
    compiled(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_lightning_aot_eager_auto_cuda_backward():
    pytest.importorskip('lightning')
    from dlshogi.ptl import Model
    from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

    model = Model(network='exp__by1x8', use_compile=True, compile_backend='aot_eager').cuda()
    assert model.model.blocks[0].attn_backend == 'auto'
    assert model.model.blocks[0]._compile_backend == 'aot_eager'
    x1 = torch.randn(2, FEATURES1_NUM, 9, 9, device='cuda')
    x2 = torch.randn(2, FEATURES2_NUM, 9, 9, device='cuda')
    sum(output.square().mean() for output in model(x1, x2)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_dynamo_onnx_unfused_auto_uses_math(tmp_path):
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxscript')
    block = AttentionBlock(192, 4).cuda().eval()
    x = torch.randn(1, 192, 9, 9, device='cuda')
    program = torch.onnx.export(block, (x,), tmp_path / 'attention.onnx', dynamo=True)
    graph = program.model_proto
    onnx.checker.check_model(graph)
    ops = [node.op_type for node in graph.graph.node]
    assert ops.count('MatMul') == 2
    assert ops.count('Softmax') == 1
    assert not block._fused and block.attn_backend == 'auto'
    with torch.no_grad():
        torch.testing.assert_close(program.exported_program.module()(x), block(x),
                                   atol=2e-5, rtol=2e-5)

