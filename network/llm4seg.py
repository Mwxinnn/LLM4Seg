import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_2d_sin_cos_positional_encoding(height, width):
    token = height * width
    # 2D coordinates and Calculate sin and cos encodings
    sin_y = torch.sin(torch.arange(token).view(1, token, 1))  # Shape (H*W, 1)
    cos_x = torch.cos(torch.arange(token).view(1, token, 1))  # Shape (H*W, 1)
    return sin_y.cuda(), cos_x.cuda()


def get_llama(h=16, w=16, layer=14):
    sin, cos = generate_2d_sin_cos_positional_encoding(h, w)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    print(f"meta-llama/Llama-3.2-1B Layer: {layer}")
    return sin, cos, model.model.layers[layer]

def get_deepseek(h=16, w=16, layer=27):
    sin, cos = generate_2d_sin_cos_positional_encoding(h, w)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    print(f"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B layer: {layer}")
    return sin, cos, model.model.layers[layer]


class LLM4Seg(nn.Module):
    def __init__(self, channel, layer, unfreeze=False, need_init=False,  h=16, w=16, mode="llama"):
        """
        Args:
            channels : input channel.
            layer: output channel.
            h: feature map height
            w: feature map weight
            mode: LLaMA or DeepSeek
        """
        super(LLM4Seg, self).__init__()
        # Encoder
        if mode == "llama":
            self.adapter1 = nn.Linear(channel, 2048)
            self.sin, self.cos, self.llm = get_llama(h, w, layer)
            self.adapter2 = nn.Linear(2048, channel)
        elif mode == "deepseek":
            self.adapter1 = nn.Linear(channel, 1536)
            self.sin, self.cos, self.llm = get_deepseek(h, w, layer)
            self.adapter2 = nn.Linear(1536, channel)
        
        
        if unfreeze:
            print(f"unFreeze {mode}")
        else:
            for param in self.llm.parameters():
                param.requires_grad = False
            print(f"Freeze {mode}")
            
        if need_init:
            print(f"Random Init weight")
            self.init_transformer_weights(self.llm)
        else:
            print(f"{mode} Init weight")
            
    def init_transformer_weights(self, module):
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                init.xavier_uniform_(submodule.weight)
                if submodule.bias is not None:
                    init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.Conv2d):
                init.kaiming_normal_(submodule.weight, mode='fan_out', nonlinearity='relu')
                if submodule.bias is not None:
                    init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.LayerNorm):
                init.constant_(submodule.weight, 1.0)
                init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.Embedding):
                init.normal_(submodule.weight, mean=0, std=0.01)



    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.adapter1(x)
        x_llm = self.llm(hidden_states=x, position_embeddings=[self.sin, self.cos])[0]
        x = self.adapter2(x_llm)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
