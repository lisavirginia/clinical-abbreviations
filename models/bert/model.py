import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import RobertaConfig, RobertaModel

class MatchHead(nn.Module):
    """Roberta Head for Matching."""
    def __init__(
        self,
        base_model_feature_size,
        num_classes,
        rnn_dimension,
        linear_1_dimension,
    ):
        """Model architecture definition for the capitalization model in torch."""
        super(MatchHead, self).__init__()
        self.GRU_1 = nn.GRU(base_model_feature_size, rnn_dimension, bidirectional=True)
        self.GRU = nn.GRU(base_model_feature_size, rnn_dimension, bidirectional=True)
        self.linear_1 = nn.Linear(rnn_dimension * 2, linear_1_dimension)
        self.linear_2 = nn.Linear(linear_1_dimension, num_classes)

    def forward(self, data_1, data_2):
        """Forward pass"""

        # batch second is faster
        features_1 = data_1.permute(1, 0, 2)
        features_2 = data_2.permute(1, 0, 2)

        gru_1_output, _ = self.GRU_1(features_1)
        gru_2_output, _ = self.GRU_2(features_2)
        
        gru_1_output_permute = gru_1_output.permute(1, 0, 2)
        gru_2_output_permute = gru_2_output.permute(1, 0, 2)

        final_gru_state_1 = torch.squeeze(gru_1_output_permute[:, -1:, :])
        final_gru_state_2 = torch.squeeze(gru_2_output_permute[:, -1:, :])
        # Undoing the above permutation now that we are through GRU
        
        linear_input = torch.cat((final_gru_state_1, final_gru_state_2), -1)
        
        linear_output = self.linear_1(linear_input)
        activated_linear_output = F.relu(linear_output)
        pre_sigmoid_output = self.linear_2(activated_linear_output)
        sigmoid_output = F.sigmoid(pre_sigmoid_output)

        return sigmoid_output

class MatchArchitecture(nn.Module):
    "Transformer base model for matching."
    def __init__(
        self,
        base_model_path,
        base_model_name,
        is_custom_pretrained,
        base_model_feature_size,
        num_classes,
        rnn_dimension,
        linear_1_dimension,
    ):
        super(MatchArchitecture, self).__init__()
        if not is_custom_pretrained:
            self.base_model = RobertaModel.from_pretrained(base_model_name)
        else:
            self.base_model = RobertaModel.from_pretrained(base_model_path)
        self.match_head = MatchHead(
            base_model_feature_size, num_classes, rnn_dimension, linear_1_dimension
        )

    def forward(
            self,
            input_ids_1,
            input_ids_2,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
    ):
        """Forward pass"""
        outputs_1 = self.base_model(
            input_ids_1,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        outputs_2 = self.base_model(
            input_ids_2,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        # Outputs[0] is seq output, outputs[1] is pooled if you want to do a seq level task
        sequence_output_1 = outputs_1[0]
        sequence_output_2 = outputs_2[0] 
        
        match_classification = self.ner_head(sequence_output_1, sequence_output_2)

        return match_classification


