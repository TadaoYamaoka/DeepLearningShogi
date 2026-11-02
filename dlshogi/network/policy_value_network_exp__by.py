import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import fuse_conv_bn_eval


from dlshogi.common import *


# -----------------------------------------------------------
# 学習時の Attention 実装 (ONNX エクスポート / fuse 後は常に "math")
#   "auto": Inductor は "flex"、その他の backend / eager 実行は "sdpa" (デフォルト)
#   "sdpa": F.scaled_dot_product_attention (相対位置バイアスを attn_mask で渡す
#           → memory-efficient / cuDNN backend)。Triton 不要。
#           ※ Inductor でコンパイルすると、Inductor がバイアスのメモリ配置を
#             並べ替え、efficient attention の backward が
#             "(*bias): last dimension must be contiguous" で失敗する。
#             eager または Inductor 以外の backend 用。
#   "flex": FlexAttention (score_mod でバイアス加算)。Inductor (Triton) が必要。
#   "math": 明示的な matmul -> softmax -> matmul (TensorRT の fused MHA パターン)
# fused backend は [B, H, S, S] の attention 行列を保存しない (backward で再計算)
# ため、GPU メモリが減り、メモリ帯域も節約できる。
# -----------------------------------------------------------
ATTN_BACKENDS = ("auto", "sdpa", "flex", "math")

try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    # flex_attention は torch.compile しないと fused kernel にならない。
    # torch.compile は遅延評価なので、ここでは実際のコンパイルは走らない。
    _compiled_flex_attention = torch.compile(_flex_attention, dynamic=False)
except ImportError:
    _compiled_flex_attention = None


class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


def _swiglu_hidden(d_model, multiple_of):
    # 通常の 4x FFN と同程度の規模になる 8/3 倍の幅を multiple_of に切り上げる。
    hidden = int(8 * d_model / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


@torch.no_grad()
def fuse_bn_conv_eval(bn, conv):
    """BN -> 1x1 Conv を 1 つの Conv (bias 付き) に畳み込む。

    推論時の BN は y = x * s + t (チャンネルごとのアフィン変換) なので
        conv(y) = (W * s) x + W t (+ b)
    となる。カーネルが 1x1 (パディングなし) の場合のみ厳密に等価。
    """
    assert not bn.training and not conv.training, "eval() してから fuse すること"
    assert conv.kernel_size == (1, 1) and conv.padding == (0, 0) and conv.groups == 1
    s = bn.weight * torch.rsqrt(bn.running_var + bn.eps)
    t = bn.bias - bn.running_mean * s

    fused = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=1, bias=True,
                      device=conv.weight.device, dtype=conv.weight.dtype).eval()
    fused.weight.copy_(conv.weight * s.view(1, -1, 1, 1))
    bias = conv.weight.flatten(1) @ t
    if conv.bias is not None:
        bias = bias + conv.bias
    fused.bias.copy_(bias)
    return fused


class MatrixConv1x1(nn.Module):
    """1x1 Conv with independent 2D parameters for matrix-aware optimizers (Muon etc.).

    Split matrices are concatenated for a single convolution. Old Conv2d
    checkpoints are accepted; inference fusion materializes a normal Conv2d.
    """

    def __init__(self, in_channels, out_channels, splits=1):
        super().__init__()
        assert out_channels % splits == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(out_channels // splits, in_channels))
            for _ in range(splits)
        ])
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=5 ** 0.5)

    def _conv_weight(self):
        weight = torch.cat(list(self.weights), dim=0) if len(self.weights) > 1 else self.weights[0]
        return weight[:, :, None, None]

    def forward(self, x):
        return F.conv2d(x, self._conv_weight())

    @torch.no_grad()
    def to_conv2d(self):
        weight = self.weights[0]
        conv = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False,
                         device=weight.device, dtype=weight.dtype)
        conv.weight.copy_(self._conv_weight())
        return conv.train(self.training)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        old_key = prefix + "weight"
        if old_key in state_dict:
            weight = state_dict.pop(old_key)
            if weight.shape == (self.out_channels, self.in_channels, 1, 1):
                weight = weight[:, :, 0, 0]
            if weight.shape != (self.out_channels, self.in_channels):
                raise RuntimeError(f"Invalid legacy convolution shape for {old_key}: {weight.shape}")
            for i, chunk in enumerate(weight.chunk(len(self.weights), dim=0)):
                state_dict[prefix + f"weights.{i}"] = chunk
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class AttentionBlock(nn.Module):
    """Peri-norm (pre-norm + post-norm) の Attention + SwiGLU FFN ブロック。

        x = x + PostBN(Attn(PreBN(x)))
        x = x + PostBN(FFN(PreBN(x)))

    SwiGLU(x) = W_out(SiLU(W_g x) * (W_v x))。
    PreBN は直後の 1x1 conv に、PostBN は直前の 1x1 conv に、
    fuse() で畳み込めるので推論コストは 0。

    テンソルのレイアウト (S = 81 マス, D = depth, H = nhead, C = d_model):
      qkv_gate (1x1 conv)  : [B, 4C, 9, 9]
      Q : [B, H, D, S] -> transpose -> [B, H, S, D]
      K : [B, H, D, S]  (NCHW の並びがそのまま K^T なので transpose 不要)
      V : [B, H, D, S] -> transpose -> [B, H, S, D]
      scores = Q @ K^T + rel_bias        : [B, H, S, S]
      out    = Softmax(scores) @ V       : [B, H, S, D] -> [B, C, 9, 9]

    推論 / ONNX エクスポート時は BMM1 -> (bias 加算) -> Softmax -> BMM2 という
    標準の並びで、TensorRT の fused MHA (_gemm_mha_v2) のパターンに一致する。
    学習時 (CUDA, fuse 前) は attn_backend の fused kernel で同じ計算をする。
    """

    SEQ_LEN = 81

    def __init__(self, d_model, nhead, ffn_multiple_of=64, attn_backend="auto"):
        super(AttentionBlock, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert attn_backend in ATTN_BACKENDS, f"attn_backend must be one of {ATTN_BACKENDS}"
        if attn_backend == "flex" and _compiled_flex_attention is None:
            raise RuntimeError("FlexAttention is not available in this PyTorch")
        self.attn_backend = attn_backend
        self._compile_backend = None
        self.d_model = d_model
        self.nhead = nhead
        self.depth = d_model // nhead
        self.scale = self.depth**-0.5
        self._fused = False

        # -----------------------------------------------------------
        # Attention
        # -----------------------------------------------------------
        self.attn_norm = nn.BatchNorm2d(d_model)  # pre-norm

        # Q, K, V と Gate (G1: SDPA 出力への elementwise gating) を
        # 1 回の 1x1 conv (GEMM) でまとめて計算する
        self.qkv_gate = MatrixConv1x1(d_model, 4 * d_model, splits=4)

        # RMS-based QK normalization (per head, over depth axis)
        # 全 head 共有の重みは no_weight_decay() で weight decay から除外する。
        self.qk_norm_eps = 1e-6
        self.q_norm_weight = nn.Parameter(torch.ones(self.depth))
        self.k_norm_weight = nn.Parameter(torch.ones(self.depth))

        # Relative Positional Encoding Setup
        window_size = 9
        self.num_relative_distance = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(nhead, self.num_relative_distance)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index_flat", relative_position_index.view(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.proj = MatrixConv1x1(d_model, d_model)
        self.attn_post_norm = nn.BatchNorm2d(d_model)  # post-norm

        # -----------------------------------------------------------
        # SwiGLU FFN: gate/value を 1 つの conv で計算する。
        # -----------------------------------------------------------
        self.ffn_norm = nn.BatchNorm2d(d_model)  # pre-norm
        self.ffn_hidden = _swiglu_hidden(d_model, ffn_multiple_of)
        self.ffn_in = MatrixConv1x1(d_model, 2 * self.ffn_hidden, splits=2)
        self.ffn_out = MatrixConv1x1(self.ffn_hidden, d_model)
        self.ffn_post_norm = nn.BatchNorm2d(d_model)  # post-norm

    def _rms_norm(self, x, dim):
        # 各 head / マスのベクトルを depth 軸で正規化する。
        # BatchNorm ではないので、バッチ統計もマス間の混合もない。
        return x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + self.qk_norm_eps)

    def _relative_position_bias(self):
        # 定数テーブル + 定数インデックスの Gather なので、ONNX エクスポート時に
        # 定数畳み込みされ、推論時は [1, H, S, S] の定数加算になる。
        bias = self.relative_position_bias_table[:, self.relative_position_index_flat]
        return bias.view(1, self.nhead, self.SEQ_LEN, self.SEQ_LEN)

    def set_compile_backend(self, backend):
        self._compile_backend = backend

    def _attention_impl(self, x):
        # CPU / fused inference / ONNX always use the explicit TensorRT pattern.
        if (self.attn_backend == "math" or self._fused or not x.is_cuda
                or torch.onnx.is_in_onnx_export()):
            return "math"
        if self.attn_backend != "auto":
            return self.attn_backend
        # Inductor 下では SDPA の bias 配置が backward を壊すため flex を使う。
        # その他の backend は flex の backward が失敗しうるため SDPA にする。
        # None は通常の torch.compile の既定 backend (Inductor) として扱う。
        if (torch.compiler.is_compiling() and self._compile_backend in (None, "inductor")
                and _compiled_flex_attention is not None):
            return "flex"
        return "sdpa"

    def _fused_attention(self, q, k, v, impl):
        """q, k, v: [B, H, S, D] -> [B, H, S, D]

        スケールは q 側に適用済みなので scale=1.0 を渡す。
        """
        # QK norm (rsqrt) は autocast で fp32 になるので v (bf16/fp16) に揃え、
        # 同時に last dim を連続にする (copy は 1 回)。
        dtype = v.dtype
        q = q.to(dtype=dtype, memory_format=torch.contiguous_format)
        k = k.to(dtype=dtype, memory_format=torch.contiguous_format)
        v = v.contiguous()
        bias = self._relative_position_bias()  # [1, H, S, S]

        if impl == "flex":
            bias = bias[0]  # [H, S, S]

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + bias[h, q_idx, kv_idx]

            return _compiled_flex_attention(q, k, v, score_mod=score_mod, scale=1.0)

        # memory-efficient kernel は attn_mask の行ストライドが 16 要素の倍数である
        # 必要があるので、81 -> 96 にパディングしてからスライスする (値は 81 のまま)。
        bias = bias.to(dtype)
        pad = (-self.SEQ_LEN) % 16
        if pad:
            bias = F.pad(bias, (0, pad))[..., : self.SEQ_LEN]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)

    @torch.no_grad()
    def fuse(self):
        """推論用に pre/post の BN をすべて隣接する 1x1 conv に畳み込む。

        fuse 後の BN は nn.Identity に置き換わるので、forward は分岐なしで
        そのまま使える。不可逆なので学習には使わないこと。
        """
        if self._fused:
            return self
        for name in ("qkv_gate", "proj", "ffn_in", "ffn_out"):
            setattr(self, name, getattr(self, name).to_conv2d())
        # pre-norm: BN -> conv
        self.qkv_gate = fuse_bn_conv_eval(self.attn_norm, self.qkv_gate)
        self.ffn_in = fuse_bn_conv_eval(self.ffn_norm, self.ffn_in)
        # post-norm: conv -> BN
        self.proj = fuse_conv_bn_eval(self.proj, self.attn_post_norm)
        self.ffn_out = fuse_conv_bn_eval(self.ffn_out, self.ffn_post_norm)
        self.attn_norm = nn.Identity()
        self.ffn_norm = nn.Identity()
        self.attn_post_norm = nn.Identity()
        self.ffn_post_norm = nn.Identity()
        self._fused = True
        return self

    def forward(self, x):
        C, H, D, S = self.d_model, self.nhead, self.depth, self.SEQ_LEN

        # -----------------------------------------------------------
        # Attention
        # -----------------------------------------------------------
        q, k, v, g = self.qkv_gate(self.attn_norm(x)).split(C, dim=1)

        # [B, C, 9, 9] -> [B, H, D, S]
        q = q.view(-1, H, D, S).transpose(2, 3)  # [B, H, S, D]
        v = v.view(-1, H, D, S).transpose(2, 3)  # [B, H, S, D]

        # QK norm (depth 軸) とスケールは BMM1 の前に Q/K 側で済ませる
        # Q 側は weight と scale をまとめて 1 回の定数乗算にする
        q = self._rms_norm(q, dim=-1) * (self.q_norm_weight * self.scale)

        impl = self._attention_impl(x)
        if impl != "math":
            # ---- 学習用: fused kernel (attention 行列を保存しない) ----
            k = k.view(-1, H, D, S).transpose(2, 3)  # [B, H, S, D]
            k = self._rms_norm(k, dim=-1) * self.k_norm_weight
            out = self._fused_attention(q, k, v, impl)     # [B, H, S, D]
        else:
            # ---- 推論 / ONNX 用 fused MHA パターン: BMM1 -> add bias -> Softmax -> BMM2 ----
            k = k.view(-1, H, D, S)                  # [B, H, D, S] = K^T
            k = self._rms_norm(k, dim=2) * self.k_norm_weight.view(1, 1, D, 1)
            scores = torch.matmul(q, k) + self._relative_position_bias()  # [B, H, S, S]
            attention = F.softmax(scores, dim=-1)
            out = torch.matmul(attention, v)                              # [B, H, S, D]
        # ------------------------------------------------------------------

        out = out.transpose(2, 3).reshape(-1, C, 9, 9)

        # Gating (G1): Y' = Y * sigmoid(X W_theta)
        out = out * torch.sigmoid(g)

        x = x + self.attn_post_norm(self.proj(out))

        # -----------------------------------------------------------
        # FFN (SwiGLU)
        # -----------------------------------------------------------
        h = self.ffn_in(self.ffn_norm(x))
        gate, value = h.split(self.ffn_hidden, dim=1)
        h = F.silu(gate) * value

        return x + self.ffn_post_norm(self.ffn_out(h))


class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks, channels, activation=nn.ReLU(), fcl=256, nhead=4, attn_backend="auto"):
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
        self._input1_fused = False
        self.norm1_2 = nn.BatchNorm2d(channels)
        self.act = activation

        # -----------------------------------------------------------
        # Absolute Positional Encoding Setup
        # -----------------------------------------------------------
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, channels, 9, 9))
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        # 中間層はすべて AttentionBlock
        self.blocks = nn.Sequential(
            *[AttentionBlock(channels, nhead, attn_backend=attn_backend) for _ in range(blocks)]
        )
        # 残差経路は正規化されないので、ヘッドの前で最終正規化する。
        # 直後は活性化なしで 1x1 conv (policy / value_conv1) なので、
        # fuse() で両ヘッドの conv に畳み込める。
        self.final_norm = nn.BatchNorm2d(channels)

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

    # weight decay から除外する位置エンコーディング系パラメータ (名前の末尾で判定)
    _NO_WEIGHT_DECAY_SUFFIXES = ("absolute_pos_embed", "relative_position_bias_table")

    def matrix_parameters(self):
        """Full names of hidden attention/FFN matrices.

        Routed by ``--param_optimizer matrix=...`` (dlshogi.optim.build_optimizer)
        to a matrix-aware optimizer such as Muon.
        """
        return {
            f"{name}.weights.{i}"
            for name, module in self.named_modules()
            if isinstance(module, MatrixConv1x1)
            for i in range(len(module.weights))
        }

    def no_weight_decay(self):
        """weight decay から除外するパラメータの完全名を返す。

        dlshogi.optim.build_optimizer() から呼ばれる。
        - 1 次元以下のパラメータ: BN の weight/bias, QK-norm の weight,
          Linear の bias, policy_bias
        - 位置エンコーディング: absolute_pos_embed, relative_position_bias_table
        正規化のスケールやバイアス、位置エンコーディングを減衰させない。
        """
        return {
            name
            for name, param in self.named_parameters()
            if param.ndim <= 1 or name.endswith(self._NO_WEIGHT_DECAY_SUFFIXES)
        }

    @torch.no_grad()
    def fuse(self):
        """推論用に BN を conv へ畳み込む。

        - 入力層: 7x7 / 1x1 の Conv-BN 2 系統を 1 つの 7x7 conv に統合
        - AttentionBlock: pre-norm (BN->conv) / post-norm (conv->BN) を畳み込み
        - ヘッド: final_norm を policy / value_conv1 に、value_norm1 を
          value_conv1 に畳み込み、policy conv の bias は policy_bias に統合
        これにより fuse 後のモデルには BatchNorm が残らない。

        学習済みチェックポイントを読み込み、eval() してから呼ぶこと。
        不可逆なので学習やチェックポイント読み込みには fuse 前のモデルを使う。
        2 回目以降の呼び出しは何もしない。
        """
        if self.training:
            raise RuntimeError("fuse() requires eval() before fusion")
        if self._input1_fused:
            return self

        # Fold each BN using its running statistics, retaining device/dtype.
        fused_7x7 = fuse_conv_bn_eval(self.l1_1_1, self.norm1_1_1)
        fused_1x1 = fuse_conv_bn_eval(self.l1_1_2, self.norm1_1_2)

        # A 1x1 kernel is a 7x7 kernel with zeros outside the center.
        fused_7x7.weight[:, :, 3:4, 3:4].add_(fused_1x1.weight)
        fused_7x7.bias.add_(fused_1x1.bias)

        self.l1_1_1 = fused_7x7
        del self.l1_1_2
        del self.norm1_1_1
        del self.norm1_1_2

        for block in self.blocks:
            block.fuse()

        # policy head: final_norm -> policy conv
        # conv の bias (出力チャンネルごと) は 81 マス分に展開して policy_bias に足す。
        # flatten(1) は [C, 9, 9] をチャンネル優先で並べるので repeat_interleave で一致する。
        policy = fuse_bn_conv_eval(self.final_norm, self.policy)
        self.policy_bias.bias.add_(policy.bias.repeat_interleave(9 * 9))
        policy.bias = None
        self.policy = policy

        # value head: final_norm -> value_conv1 -> value_norm1
        value_conv1 = fuse_bn_conv_eval(self.final_norm, self.value_conv1)
        self.value_conv1 = fuse_conv_bn_eval(value_conv1, self.value_norm1)

        self.final_norm = nn.Identity()
        self.value_norm1 = nn.Identity()

        self._input1_fused = True
        return self

    def forward(self, x1, x2):
        if self._input1_fused:
            u1_1 = self.l1_1_1(x1)
        else:
            u1_1_1 = self.norm1_1_1(self.l1_1_1(x1))
            u1_1_2 = self.norm1_1_2(self.l1_1_2(x1))
            u1_1 = u1_1_1 + u1_1_2
        u1_2 = self.norm1_2(self.l1_2(x2))
        u1 = self.act(u1_1 + u1_2 + self.absolute_pos_embed)

        # attention blocks
        h = self.blocks(u1)
        h = self.final_norm(h)

        # policy network
        h_policy = self.policy(h)
        h_policy = self.policy_bias(torch.flatten(h_policy, 1))

        # value network
        h_value = self.act(self.value_norm1(self.value_conv1(h)))
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value
