from __future__ import annotations

import functools
import math
from typing import Callable, Optional, Tuple, Union

import torch



def init_method_normal(mean: float = 0.0, scale: float = 1.0) -> Callable[[torch.Tensor],torch.Tensor]:
    """Initialize tensor with normal distribution."""
    return functools.partial(torch.nn.init.normal_, mean=mean, std=scale)
    
def init_method_semi_orthogonal(
    scale: float = 1.0
) -> Callable[[torch.Tensor],torch.Tensor]:
    """Initialize tensor with semi-orthogonal distribution."""
    def init(w, scale=scale) -> torch.Tensor:
        w_fp = w.data.double()
        torch.nn.init.orthogonal_(w_fp, gain=scale)
        w.data = w_fp.to(dtype=w.data.dtype)
        return w
    return functools.partial(init, scale=scale)

def init_method_col_norm(scale: float = 1.0, is_input: bool = True) -> Callable[[torch.Tensor],torch.Tensor]:
    """Initialize tensor with column norm."""
    def init(w: torch.Parameter, scale=scale, is_input:bool=True) -> torch.Tensor:
        dtype = w.data.dtype
        if is_input:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=0, keepdim=True)
        # tensor *= math.sqrt(tensor.size(0))
        w.data *= scale
        # if self.normalized:
        #     w.data /= w.size(1)
        w.data = w.data.to(dtype=dtype)
        if is_input:
            w.data = w.data.transpose(0, 1)
        return w
    return functools.partial(init, scale=scale, is_input=is_input)
    
def init_method_row_norm(scale: float = 1.0, is_input: bool = True) -> Callable[[torch.Tensor],torch.Tensor]:
    """Initialize tensor with row norm."""
    def init(w: torch.Parameter, scale=scale, is_input:bool=True) -> torch.Parameter:
        dtype = w.data.dtype
        if is_input:
            tensor = tensor.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=-1, keepdim=True)
        # tensor *= math.sqrt(tensor.size(0))
        w.data *= scale
        # if self.normalized:
        #     w.data /= w.size(1)
        w.data = w.data.to(dtype=dtype)
        if is_input:
            w.data = w.data.transpose(0, 1)
        return w
    return functools.partial(init, scale=scale, is_input=is_input)



def init_scheme_from_args(args) -> dict:
    init_scheme = args.initialization_scheme

    kwargs_update = dict()
    init_method=None
    input_init_method=None
    output_init_method=None
    output_layer_init_method=None
    mlp_init_method=None
    mlp_out_init_method=None
    hidden_init_method=None
    # init_method_std=args.init_method_std,
    input_init_method_scale=args.init_input_scale
    output_init_method_scale=args.init_output_scale
    output_layer_init_method_scale=args.init_projection_scale
    mlp_init_method_scale=args.init_mlp1_scale
    mlp_out_init_method_scale=args.init_mlp2_scale
    hidden_init_method_scale=args.init_hidden_scale
        # )

    embedding_size = args.hidden_size
    if args.ffn_hidden_size is not None:
        ffn_size = args.ffn_hidden_size
        ffn_1_size = ffn_size
    elif args.gated_linear_unit or args.swiglu:
        ffn_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        ffn_1_size = 2 * ffn_size
    else:
        ffn_size = args.hidden_size
        ffn_1_size = ffn_size

    qkv_out_size = args.hidden_size * 3

    if init_scheme == 'normal':
        init_method = None
    elif init_scheme == 'megatron':
        print("Type: ", type(args.num_layers))
        print(args.num_layers)
        projection_init_scale = 1.0 / math.sqrt(2.0 * args.num_layers)
        print("Type: ", type(projection_init_scale))
        print("Type: ", type(output_layer_init_method_scale))
        init_method = None
        output_layer_init_method_scale *= projection_init_scale
        mlp_out_init_method_scale *= projection_init_scale
    elif init_scheme == "gpt-neox":
        neox_init_scale = math.sqrt(2.0 / (5.0 * args.hidden_size))
        neox_projection_init_scale = 1.0 / math.sqrt(args.hidden_size) / args.num_layers

        init_method = init_method_normal(0.0, scale=args.init_method_std * neox_init_scale)
        output_layer_init_method_scale *= neox_projection_init_scale
        mlp_out_init_method_scale *= neox_projection_init_scale
        input_init_method_scale *= neox_init_scale
        hidden_init_method_scale *= neox_init_scale
        mlp_init_method_scale *= neox_init_scale
        output_init_method_scale *= neox_init_scale
    # elif init_scheme == "full-gpt_neox":
    #     neox_init_scale = math.sqrt(2.0 / (5.0 * args.hidden_size))
    #     neox_projection_init_scale = 1.0 / math.sqrt(args.hidden_size) / args.num_layers

    #     init_method = init_method_normal(0.0, scale=neox_init_scale)
    #     output_layer_init_scale *= neox_projection_init_scale
    #     mlp_out_init_method_scale *= neox_projection_init_scale
    #     input_init_method_scale *= neox_init_scale
    #     hidden_init_method_scale *= neox_init_scale
    #     mlp_init_method_scale *= neox_init_scale
    #     output_init_method_scale *= neox_init_scale
    
    elif init_scheme == "scaled":
        neox_init_scale = math.sqrt(2.0 / (5.0 * args.hidden_size))
        scaled_neox_projection_init_scale = neox_init_scale / math.sqrt(args.num_layers * 2.0)

        init_method = init_method_normal(0.0, scale=neox_init_scale)
        output_layer_init_method_scale *= scaled_neox_projection_init_scale
        mlp_out_init_method_scale *= scaled_neox_projection_init_scale
        input_init_method_scale *= neox_init_scale
        hidden_init_method_scale *= neox_init_scale
        mlp_init_method_scale *= neox_init_scale
        output_init_method_scale *= neox_init_scale
    elif init_scheme == "modular":

        input_init_method = init_method_col_norm(scale=input_init_method_scale, is_input=True)
        output_init_method = init_method_row_norm(scale=output_init_method_scale, is_input=False)
        output_layer_init_method = init_method_semi_orthogonal(scale=output_layer_init_method_scale)
        # Changes due to MLP layer width changing for scion/mup variants
        mlp_init_method = init_method_semi_orthogonal(scale=mlp_init_method_scale)
        mlp_out_init_method = init_method_semi_orthogonal(scale=mlp_out_init_method_scale)
        hidden_init_method = init_method_semi_orthogonal(scale=hidden_init_method_scale)
    elif init_scheme == "scion":
        hidden_multiplier = math.sqrt(embedding_size) 
        input_init_method_scale *= math.sqrt(embedding_size)
        # Changes due to MLP layer width changing for scion/mup variants
        mlp_init_method_scale *= math.sqrt(ffn_1_size)
        mlp_out_init_method_scale *= math.sqrt(embedding_size)

        output_layer_init_method_scale *= hidden_multiplier
        hidden_init_method_scale *= (hidden_multiplier * math.sqrt(3))
        
        input_init_method = init_method_col_norm(scale=input_init_method_scale, is_input=True)
        output_init_method = init_method_row_norm(scale=output_init_method_scale, is_input=False)
        output_layer_init_method = init_method_semi_orthogonal(scale=output_layer_init_method_scale)
        
        mlp_init_method = init_method_semi_orthogonal(scale=mlp_init_method_scale)
        mlp_out_init_method = init_method_semi_orthogonal(scale=mlp_out_init_method_scale)
        hidden_init_method = init_method_semi_orthogonal(scale=hidden_init_method_scale)
    elif init_scheme == "normed-scion":
        # hidden_multiplier = math.sqrt(embedding_size) 
        input_init_method_scale *= math.sqrt(embedding_size) /  args.padded_vocab_size
        # Changes due to MLP layer width changing for scion/mup variants
        mlp_init_method_scale *= math.sqrt(ffn_1_size / embedding_size)
        mlp_out_init_method_scale *= math.sqrt(embedding_size / ffn_size)

        # output_layer_init_method_scale *= hidden_multiplier
        hidden_init_method_scale *= math.sqrt(3)
        output_init_method_scale *= math.sqrt(args.padded_vocab_size)
        
        input_init_method = init_method_col_norm(scale=input_init_method_scale, is_input=True)
        output_init_method = init_method_row_norm(scale=output_init_method_scale, is_input=False)
        output_layer_init_method = init_method_semi_orthogonal(scale=output_layer_init_method_scale)
        
        mlp_init_method = init_method_semi_orthogonal(scale=mlp_init_method_scale)
        mlp_out_init_method = init_method_semi_orthogonal(scale=mlp_out_init_method_scale)
        hidden_init_method = init_method_semi_orthogonal(scale=hidden_init_method_scale)
    elif init_scheme == "mup":
        init_method = None
        input_init_method_scale *= 1
        mlp_out_init_method_scale *= (ffn_size**-0.5)
        mlp_init_method_scale *= (embedding_size**-0.5)
        output_layer_init_method_scale *= (embedding_size**-0.5)
        hidden_init_method_scale *= (embedding_size**-0.5)
        output_init_method_scale *= 1
        
        kwargs_update["output_multiplier"] = args.output_multiplier / embedding_size 
    elif init_scheme == "sp":
        # SP initialization scheme according to Mup paper (https://arxiv.org/abs/2203.03466)
        init_method = None
        # setting input_init_method_scale to 1 instead of 1/fan_in because fan_in here is vocab_size which does not scale with width, so any constant is fine. This is according to the Mup paper Appendix B.1 (https://arxiv.org/abs/2203.03466)
        input_init_method_scale *= 1
        hidden_init_method_scale *= (embedding_size**-0.5)
        output_layer_init_method_scale *= (embedding_size**-0.5)
        mlp_init_method_scale *= (embedding_size**-0.5)
        mlp_out_init_method_scale *= (ffn_size**-0.5)
        output_init_method_scale *= (embedding_size**-0.5)
    else:
        raise ValueError(f"Unknown initialization scheme: {init_scheme}")
    
    kwargs_update['init_method'] = init_method
    kwargs_update['input_init_method'] = input_init_method
    kwargs_update['output_init_method'] = output_init_method
    kwargs_update['output_layer_init_method'] = output_layer_init_method
    kwargs_update['mlp_init_method'] = mlp_init_method
    kwargs_update['mlp_out_init_method'] = mlp_out_init_method
    kwargs_update['hidden_init_method'] = hidden_init_method
    # kwargs_update['init_method_std'] = args.init_method_std
    kwargs_update['input_init_method_scale'] = input_init_method_scale
    kwargs_update['output_init_method_scale'] = output_init_method_scale
    kwargs_update['output_layer_init_method_scale'] = output_layer_init_method_scale
    kwargs_update['mlp_init_method_scale'] = mlp_init_method_scale
    kwargs_update['mlp_out_init_method_scale'] = mlp_out_init_method_scale
    kwargs_update['hidden_init_method_scale'] = hidden_init_method_scale

    return kwargs_update
