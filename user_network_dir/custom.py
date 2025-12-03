"""Custom recognition model for EasyOCR using benchmark architecture.

This keeps the original Transformation / FeatureExtraction / SequenceModeling /
Prediction logic, but exposes an __init__ signature compatible with
EasyOCR's get_recognizer, which calls:

    Model(num_class=num_class, **network_params)
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor,
)
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(
        self,
        num_class,
        input_channel=1,
        output_channel=256,
        hidden_size=256,
        imgH=32,
        imgW=100,
        Transformation="TPS",
        FeatureExtraction="VGG",
        SequenceModeling="BiLSTM",
        Prediction="CTC",
        num_fiducial=20,
        **kwargs,
    ):
        """EasyOCR-compatible constructor.

        All keyword arguments come from custom.yaml's network_params.
        """
        super(Model, self).__init__()

        # Remember stages for forward logic
        self.stages = {
            "Trans": Transformation,
            "Feat": FeatureExtraction,
            "Seq": SequenceModeling,
            "Pred": Prediction,
        }

        # ---------------------- Transformation ----------------------
        if Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial,
                I_size=(imgH, imgW),
                I_r_size=(imgH, imgW),
                I_channel_num=input_channel,
            )
        else:
            self.Transformation = None

        # ---------------------- FeatureExtraction ----------------------
        if FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        else:
            raise Exception("No FeatureExtraction module specified")

        self.FeatureExtraction_output = output_channel
        # Transform final (imgH/16-1) -> 1
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # ---------------------- Sequence modeling ----------------------
        if SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
            )
            self.SequenceModeling_output = hidden_size
        else:
            self.SequenceModeling = None
            self.SequenceModeling_output = self.FeatureExtraction_output

        # ---------------------- Prediction ----------------------
        if Prediction == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)
        elif Prediction == "Attn":
            self.Prediction = Attention(self.SequenceModeling_output, hidden_size, num_class)
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def forward(self, input, text=None, is_train=True, batch_max_length=None):
        # Transformation stage
        if self.stages["Trans"] != "None" and self.Transformation is not None:
            input = self.Transformation(input)

        # Feature extraction stage
        visual_feature = self.FeatureExtraction(input)
        # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        # Sequence modeling stage
        if self.stages["Seq"] == "BiLSTM" and self.SequenceModeling is not None:
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            # for convenience: not contextually modeled by BiLSTM
            contextual_feature = visual_feature

        # Prediction stage
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            # Attention prediction requires text and batch_max_length
            prediction = self.Prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=batch_max_length,
            )

        return prediction
