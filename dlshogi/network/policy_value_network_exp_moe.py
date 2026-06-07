import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from dlshogi.common import *


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


class DroplessMoESwiGLUFFN(nn.Module):
    """Top-2 dropless MoE FFN for NCHW feature maps.

    Each token is routed to exactly two experts. Expert matmuls are packed by
    expert id and executed with torch.nn.functional.grouped_mm. Router weights
    are intentionally kept in FP32 for stable routing under mixed precision.
    """

    top_k = 2
    grouped_mm_supported_dtypes = (torch.bfloat16,)

    def __init__(self, in_features, hidden_features, num_experts=8):
        super(DroplessMoESwiGLUFFN, self).__init__()
        if num_experts < self.top_k:
            raise ValueError("num_experts must be at least 2 when top_k is fixed to 2.")

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_experts = num_experts

        self.router = nn.Linear(in_features, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(num_experts, in_features, 2 * hidden_features))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_features, in_features))
        self.act = nn.SiLU()
        self.aux_loss = None
        self.z_loss = None

        # Non-persistent caches for routing metadata and ignored grouped_mm
        # padding rows. They are rebuilt automatically when shape/device/dtype
        # changes and are intentionally excluded from state_dict.
        self.register_buffer("_flat_token_cache", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_padding_cache", torch.empty(0), persistent=False)
        self.register_buffer("_hidden_padding_cache", torch.empty(0), persistent=False)
        self.last_expert_counts = None
        self.last_expert_load_entropy = None
        self.last_expert_max_load_ratio = None

        self._reset_parameters()

    def _reset_parameters(self):
        # 1. routerは小さめ。
        #    0初期化はtopkのtieで特定expertに固定されやすいので避ける。
        nn.init.normal_(self.router.weight, mean=0.0, std=0.005)

        # 2. w1は「入力の分散を1前後に保つ」fan-in初期化にする。
        #    x @ w1 なので fan_in = in_features。
        w1_std = 1.0 / math.sqrt(self.in_features)
        nn.init.normal_(self.w1, mean=0.0, std=w1_std)

        # 3. w2はFFN branchの初期出力をどれくらい残差に効かせたいかで決める。
        #    SiLU(N(0,1))^2 の期待値はだいたい0.356なので、
        #    value * SiLU(gate) のstdは約sqrt(0.356)=0.596。
        #
        #    residual_target_std = 0.5: 安全寄り
        #    residual_target_std = 0.7: dense BNありFFNに近づける寄り
        residual_target_std = 0.5
        swiglu_second_moment = 0.356
        w2_std = residual_target_std / math.sqrt(
            self.hidden_features * swiglu_second_moment
        )
        nn.init.normal_(self.w2, mean=0.0, std=w2_std)

    def _apply(self, fn):
        module = super()._apply(fn)
        self.router.float()
        return module

    def configure_expert_dtype(self, dtype=torch.bfloat16):
        """Optionally keep expert parameters in a storage dtype.

        PyTorch grouped_mm currently expects BF16 CUDA inputs on SM80+ GPUs.
        Use torch.bfloat16 to avoid per-forward expert-weight casts in BF16
        autocast runs. torch.float32 is accepted only as a storage/revert dtype;
        the grouped_mm operands are still cast to the BF16 activation dtype.
        FP16 is intentionally rejected unless this model is adapted and tested
        against a grouped_mm build that supports FP16.
        """
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                "dtype must be torch.bfloat16 or torch.float32. "
                "FP16 grouped_mm is not enabled in this implementation."
            )
        with torch.no_grad():
            self.w1.data = self.w1.data.to(dtype=dtype)
            self.w2.data = self.w2.data.to(dtype=dtype)
        self.router.float()
        return self

    def _check_grouped_mm_backend(self, x_tokens):
        if x_tokens.dtype not in self.grouped_mm_supported_dtypes:
            supported = ", ".join(str(dtype).replace("torch.", "") for dtype in self.grouped_mm_supported_dtypes)
            raise RuntimeError(
                f"DroplessMoESwiGLUFFN grouped_mm path expects {supported} CUDA activations, "
                f"got {x_tokens.dtype}. Run the model under CUDA BF16 autocast, "
                "for example torch.autocast(device_type='cuda', dtype=torch.bfloat16)."
            )
        major, minor = torch.cuda.get_device_capability(x_tokens.device)
        if major < 8:
            raise RuntimeError(
                "DroplessMoESwiGLUFFN grouped_mm path expects a CUDA GPU with SM>=80 "
                f"for BF16 grouped_mm, got capability sm_{major}{minor}."
            )

    def _prepare_grouped_mm_input(self, x_tokens):
        if x_tokens.dtype in self.grouped_mm_supported_dtypes:
            return x_tokens

        autocast_enabled = torch.is_autocast_enabled("cuda")
        autocast_dtype = torch.get_autocast_dtype("cuda") if autocast_enabled else None
        if autocast_enabled and autocast_dtype in self.grouped_mm_supported_dtypes and x_tokens.dtype == torch.float32:
            return x_tokens.to(dtype=autocast_dtype)

        return x_tokens

    def _get_flat_token(self, tokens, device):
        needed = tokens * self.top_k
        cache = self._flat_token_cache
        if cache.device != device or cache.numel() != needed:
            cache = torch.arange(tokens, device=device).repeat_interleave(self.top_k)
            self._flat_token_cache = cache
        return cache

    def _get_zero_row(self, width, dtype, device, cache_name):
        cache = getattr(self, cache_name)
        if cache.device != device or cache.dtype != dtype or cache.shape != (1, width):
            cache = torch.zeros(1, width, dtype=dtype, device=device)
            setattr(self, cache_name, cache)
        return cache

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError(
                "DroplessMoESwiGLUFFN requires CUDA tensors because it uses "
                "torch.nn.functional.grouped_mm."
            )
        if not hasattr(F, "grouped_mm"):
            raise RuntimeError("torch.nn.functional.grouped_mm is not available in this PyTorch build.")
        if self.router.weight.dtype != torch.float32:
            raise RuntimeError("Router weights must remain FP32 for stable routing.")

        batch_size, channels, height, width = x.shape
        if channels != self.in_features:
            raise ValueError(f"Expected {self.in_features} channels, got {channels}.")

        # grouped_mm works on token-major matrices. Make this conversion explicit:
        # the preceding permute changes strides, so contiguous()+view avoids an
        # implicit copy hidden inside reshape and guarantees contiguous C features.
        tokens = batch_size * height * width
        x_tokens = x.permute(0, 2, 3, 1).contiguous().view(tokens, channels)
        x_tokens = self._prepare_grouped_mm_input(x_tokens)
        self._check_grouped_mm_backend(x_tokens)

        # Routing is computed in FP32; expert computation keeps activation dtype.
        # In eval/inference, top-k over logits plus softmax over selected logits
        # is equivalent to top-k over full softmax followed by renormalization,
        # but avoids the full softmax and auxiliary loss statistics.
        with torch.autocast(device_type="cuda", enabled=False):
            logits = self.router(x_tokens.float())
            if self.training:
                probs = F.softmax(logits, dim=-1)
                topk_prob, topk_expert = torch.topk(probs, k=self.top_k, dim=-1)
                topk_gate = topk_prob / topk_prob.sum(dim=-1, keepdim=True)
            else:
                topk_logits, topk_expert = torch.topk(logits, k=self.top_k, dim=-1)
                topk_gate = F.softmax(topk_logits, dim=-1)

        flat_expert = topk_expert.flatten()
        flat_gate = topk_gate.flatten()
        flat_token = self._get_flat_token(tokens, x.device)

        order = torch.argsort(flat_expert)
        expert_sorted = flat_expert.index_select(0, order)
        token_sorted = flat_token.index_select(0, order)
        gate_sorted = flat_gate.index_select(0, order)
        x_sorted = x_tokens.index_select(0, token_sorted)

        counts = torch.bincount(expert_sorted, minlength=self.num_experts)
        offsets = torch.cumsum(counts, dim=0).to(torch.int32)

        # Optional logging stats for expert-load collapse checks.
        # Keep disabled by default because these extra FP32 reductions run in
        # every forward pass and are not needed for normal training/inference.
        # counts_f = counts.to(torch.float32)
        # load_prob = counts_f / counts_f.sum().clamp_min(1.0)
        # self.last_expert_counts = counts.detach()
        # self.last_expert_load_entropy = -(load_prob * load_prob.clamp_min(1e-12).log()).sum().detach()
        # self.last_expert_max_load_ratio = (counts_f.max() / counts_f.mean().clamp_min(1.0)).detach()

        if self.training:
            tokens_per_expert = counts.to(probs.dtype) / counts.sum().to(probs.dtype)
            prob_per_expert = probs.mean(dim=0)
            self.aux_loss = self.num_experts * torch.sum(tokens_per_expert * prob_per_expert)
            self.z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
        else:
            self.aux_loss = None
            self.z_loss = None

        w1 = self.w1 if self.w1.dtype == x_tokens.dtype else self.w1.to(dtype=x_tokens.dtype)

        # grouped_mm requires offs[-1] < mat_a.shape[0], so append one ignored row.
        padding = self._get_zero_row(x_sorted.shape[-1], x_sorted.dtype, x_sorted.device, "_padding_cache")
        x_grouped = torch.cat((x_sorted, padding), dim=0)
        h = F.grouped_mm(x_grouped, w1, offs=offsets)[: x_sorted.shape[0]]

        gate, value = h.split((self.hidden_features, self.hidden_features), dim=-1)
        h = value * self.act(gate)

        w2 = self.w2 if self.w2.dtype == h.dtype else self.w2.to(dtype=h.dtype)

        # grouped_mm requires offs[-1] < mat_a.shape[0], so append one ignored row.
        hidden_padding = self._get_zero_row(h.shape[-1], h.dtype, h.device, "_hidden_padding_cache")
        h_grouped = torch.cat((h, hidden_padding), dim=0)
        y_sorted = F.grouped_mm(h_grouped, w2, offs=offsets)[: h.shape[0]]
        y_sorted = y_sorted * gate_sorted.unsqueeze(-1).to(dtype=y_sorted.dtype)

        # Undo the expert-id sort back to the original flat assignment order and
        # reduce the top-k dimension explicitly. This avoids duplicate-index
        # index_add_ on CUDA, which can use atomic adds and become nondeterministic
        # or slower under deterministic execution modes.
        y_flat = torch.empty(
            tokens * self.top_k,
            channels,
            dtype=x_tokens.dtype,
            device=x_tokens.device,
        )
        y_flat.index_copy_(0, order, y_sorted.to(dtype=y_flat.dtype))
        y = y_flat.view(tokens, self.top_k, channels).sum(dim=1)
        return y.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()


class ResNetBlock(nn.Module):
    def __init__(self, channels, activation, use_se=False):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = activation
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        return self.act(out + x)


class InceptionBlock(nn.Module):
    def __init__(self, channels, activation):
        super(InceptionBlock, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=(9, 1), bias=False)
        self.conv1_2 = nn.Conv2d(channels, channels, kernel_size=(1, 9), bias=False)
        self.conv1_3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels)
        self.bn1_2 = nn.BatchNorm2d(channels)
        self.bn1_3 = nn.BatchNorm2d(channels)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=(9, 1), bias=False)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=(1, 9), bias=False)
        self.conv2_3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels)
        self.bn2_2 = nn.BatchNorm2d(channels)
        self.bn2_3 = nn.BatchNorm2d(channels)
        self.act = activation

    def forward(self, x):
        out = (
            self.bn1_1(self.conv1_1(x))
            + self.bn1_2(self.conv1_2(x))
            + self.bn1_3(self.conv1_3(x))
        )
        out = self.act(out)

        out = (
            self.bn2_1(self.conv2_1(out))
            + self.bn2_2(self.conv2_2(out))
            + self.bn2_3(self.conv2_3(out))
        )

        return self.act(out + x)


class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, activation, moe_num_experts=8):
        super(AttentionBlock, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.depth = d_model // nhead
        self.scale = self.depth**-0.5

        # -----------------------------------------------------------
        # RMS-based QK normalization
        # q/k are produced as [B, d_model, 9, 9]. For attention we
        # reinterpret them as [B, nhead, depth, 81], so RMS normalization
        # must be applied over the per-head depth axis, not over batch,
        # token, or spatial axes. This keeps each square/head vector scale
        # stable before QK^T is computed.
        # -----------------------------------------------------------
        self.qk_norm_eps = 1e-6

        # -----------------------------------------------------------
        # Gating Mechanism Implementation
        # Paper: "Gated Attention for Large Language Models"
        # Method: SDPA Elementwise Gating (G1)
        # Position: After SDPA, before Output Projection
        # -----------------------------------------------------------
        self.qkv_gate = nn.Conv2d(d_model, 4 * d_model, kernel_size=1, bias=False)

        # -----------------------------------------------------------
        # Relative Positional Encoding Setup
        # -----------------------------------------------------------
        window_size = 9
        self.num_relative_distance = (2 * window_size - 1) * (2 * window_size - 1)
        # 学習可能なバイアステーブル
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(nhead, self.num_relative_distance)
        )
        # 相対位置インデックスの生成
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        # 2点間の相対座標を計算
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index_flat", relative_position_index.view(-1))
        # パラメータ初期化（小さな分散で初期化するのが一般的）
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(d_model)

        self.act = activation

        # -----------------------------------------------------------
        # Feed-Forward Network (FFN): top-2 Dropless MoE SwiGLU
        #   hidden = 4 * d_model (same width as the previous FFN)
        #   num_experts defaults to 8 and top_k is fixed to 2.
        # -----------------------------------------------------------
        self.ffn_hidden = d_model * 4
        self.ffn = DroplessMoESwiGLUFFN(
            d_model, self.ffn_hidden, num_experts=moe_num_experts
        )

    def _rms_qk_norm(self, x):
        # x: [B, nhead, depth, 81]
        # Normalize each head/square vector over depth. This is deliberately
        # not BatchNorm: no batch statistics and no mixing across board squares.
        rms = torch.rsqrt(x.pow(2).mean(dim=2, keepdim=True) + self.qk_norm_eps)
        return x * rms

    def forward(self, x):
        qkvg = self.qkv_gate(x)
        q, k, v, g = qkvg.split((self.d_model, self.d_model, self.d_model, self.d_model), dim=1)

        # Split heads first: [B, d_model, 9, 9] -> [B, nhead, depth, 81].
        # RMS-based QK norm is applied after q/k split and over depth only.
        q = q.view(-1, self.nhead, self.depth, 81)
        k = k.view(-1, self.nhead, self.depth, 81)
        q = self._rms_qk_norm(q)
        k = self._rms_qk_norm(k)

        q = q.transpose(2, 3)
        v = v.view(-1, self.nhead, self.depth, 81)

        # -----------------------------------------------------------
        # Apply Relative Positional Encoding
        # -----------------------------------------------------------
        relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index_flat]
        relative_position_bias = relative_position_bias.view(1, self.nhead, 81, 81)

        scores = torch.matmul(q, k) * self.scale + relative_position_bias

        attention = F.softmax(scores, dim=-1)

        out = torch.matmul(v, attention.transpose(2, 3))

        out = out.view(-1, self.d_model, 9, 9)

        # -----------------------------------------------------------
        # Apply Gating
        # Formula: Y' = Y * sigmoid(X * W_theta)
        # Y is 'out' (SDPA output), X is 'x' (Block input)
        # This introduces input-dependent sparsity
        # -----------------------------------------------------------
        gating_scores = torch.sigmoid(g)
        out = out * gating_scores

        out = self.proj(out)
        out = self.bn(out)

        x = out + x

        # -----------------------------------------------------------
        # FFN (top-2 Dropless MoE SwiGLU)
        # -----------------------------------------------------------
        out = self.ffn(x)

        return self.act(out + x)


class PolicyValueNetwork(nn.Module):
    def __init__(
        self, blocks, channels, activation=nn.ReLU(), fcl=256, moe_num_experts=8
    ):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.l1_1_2 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.l1_2 = nn.Conv2d(
            in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False
        )  # pieces_in_hand
        self.norm1_1_1 = nn.BatchNorm2d(channels)
        self.norm1_1_2 = nn.BatchNorm2d(channels)
        self.norm1_2 = nn.BatchNorm2d(channels)
        self.act = activation

        # -----------------------------------------------------------
        # Absolute Positional Encoding Setup
        # -----------------------------------------------------------
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, channels, 9, 9))
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        # Resnet blocks
        _blocks = []
        for i in range(blocks):
            if i % 10 == 8:
                _blocks.append(
                    AttentionBlock(channels, 4, activation, moe_num_experts=moe_num_experts)
                )
            elif i % 10 == 9:
                pass
            elif i % 5 == 4:
                _blocks.append(InceptionBlock(channels, activation))
            elif i % 5 == 3:
                _blocks.append(ResNetBlock(channels, activation, use_se=True))
            else:
                _blocks.append(ResNetBlock(channels, activation))
        self.blocks = nn.Sequential(*_blocks)
        self._last_moe_loss = None
        self._last_moe_z_loss = None

        # policy network
        self.policy = nn.Conv2d(
            in_channels=channels,
            out_channels=MAX_MOVE_LABEL_NUM,
            kernel_size=1,
            bias=False,
        )
        self.policy_bias = Bias(9 * 9 * MAX_MOVE_LABEL_NUM)

        # value network
        self.value_conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=MAX_MOVE_LABEL_NUM,
            kernel_size=1,
            bias=False,
        )
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def moe_loss(self):
        return self._last_moe_loss

    def moe_z_loss(self):
        return self._last_moe_z_loss

    @staticmethod
    def _mean_or_none(losses):
        if not losses:
            return None
        return torch.stack(losses).mean()

    def configure_moe_expert_dtype(self, dtype=torch.bfloat16):
        """Optionally cast MoE expert parameters while keeping routers FP32.

        This is intended for AMP/BF16 runs where repeated forward-time expert
        weight casts are visible in profiling. Call before creating the
        optimizer so optimizer state matches the chosen expert dtype.
        """
        for block in self.blocks:
            if isinstance(block, AttentionBlock):
                block.ffn.configure_expert_dtype(dtype)
        return self

    def forward(self, x1, x2, return_moe_losses=False):
        u1_1_1 = self.norm1_1_1(self.l1_1_1(x1))
        u1_1_2 = self.norm1_1_2(self.l1_1_2(x1))
        u1_2 = self.norm1_2(self.l1_2(x2))
        u1 = self.act(u1_1_1 + u1_1_2 + u1_2 + self.absolute_pos_embed)

        # resnet blocks. Iterate explicitly so the MoE auxiliary losses are
        # collected from this exact forward pass, instead of being discovered
        # later by scanning module state that may be stale after re-entrant or
        # multiple forwards. The legacy moe_loss()/moe_z_loss() accessors remain
        # available and return these last collected values.
        h = u1
        moe_losses = []
        moe_z_losses = []
        for block in self.blocks:
            h = block(h)
            if isinstance(block, AttentionBlock):
                if block.ffn.aux_loss is not None:
                    moe_losses.append(block.ffn.aux_loss)
                if block.ffn.z_loss is not None:
                    moe_z_losses.append(block.ffn.z_loss)
        self._last_moe_loss = self._mean_or_none(moe_losses)
        self._last_moe_z_loss = self._mean_or_none(moe_z_losses)

        # policy network
        h_policy = self.policy(h)
        h_policy = self.policy_bias(torch.flatten(h_policy, 1))

        # value network
        h_value = self.act(self.value_norm1(self.value_conv1(h)))
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        if return_moe_losses:
            return h_policy, h_value, self._last_moe_loss, self._last_moe_z_loss
        return h_policy, h_value

    def forward_with_moe_losses(self, x1, x2):
        return self.forward(x1, x2, return_moe_losses=True)
