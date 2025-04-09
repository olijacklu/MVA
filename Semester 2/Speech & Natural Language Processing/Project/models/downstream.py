import torch
import torch.nn as nn
from peft import get_peft_model


class DownstreamModel(nn.Module):
    def __init__(self, upstream_model, num_chars=None, num_languages=None, task="asr", lora_config=None):
        super().__init__()

        self.task = task

        self.upstream_model = get_peft_model(upstream_model, lora_config) if lora_config else upstream_model

        for param_name, param in self.upstream_model.named_parameters():
            if lora_config and 'lora' not in param_name:
                param.requires_grad = False
            elif not lora_config:
                param.requires_grad = False

        self.layer_weights = nn.Parameter(torch.ones(upstream_model.config.num_hidden_layers) / upstream_model.config.num_hidden_layers)

        self.conv_downsample = nn.Conv1d(
            upstream_model.config.hidden_size,
            upstream_model.config.hidden_size,
            kernel_size=2,
            stride=2
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=upstream_model.config.hidden_size,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        if task in ["asr", "asr+lid"]:
            self.asr_classifier = nn.Linear(upstream_model.config.hidden_size, num_chars)

        if task in ["lid", "asr+lid"] and num_languages is not None:
            self.lid_classifier = nn.Linear(upstream_model.config.hidden_size, num_languages)

    def forward(self, input_values, input_lengths=None):
        input_values = input_values.to(torch.float16)

        with torch.no_grad():
            if hasattr(self.upstream_model, 'base_model'):
                outputs = self.upstream_model.base_model(
                    input_values,
                    output_hidden_states=True,
                    return_dict=True
                )
            else:
                outputs = self.upstream_model(
                    input_values,
                    output_hidden_states=True,
                    return_dict=True
                )

        hidden_states = outputs.hidden_states[1:]
        weighted_sum = torch.zeros_like(hidden_states[0])

        for i, h in enumerate(hidden_states):
            weighted_sum += self.layer_weights[i] * h

        x = weighted_sum.transpose(1, 2)
        x = self.conv_downsample(x)
        x = x.transpose(1, 2)

        x = self.transformer(x)

        result = {}

        if self.task in ["asr", "asr+lid"]:
            result["asr_logits"] = self.asr_classifier(x)

        if self.task in ["lid", "asr+lid"]:
            pooled = torch.mean(x, dim=1)
            result["lid_logits"] = self.lid_classifier(pooled)

        if self.task == "asr":
            return result["asr_logits"]

        return result
