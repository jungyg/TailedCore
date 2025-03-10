

import torch
import torch.nn.functional as F
from typing import List

from .patch_maker import PatchMaker
from .feature_aggregator import NetworkFeatureAggregator
from .common import Preprocessing, Aggregator
from .backbone import get_backbone

class FeatureExtractor(torch.nn.Module):

    def __init__(
        self,
        input_shape,
        backbone_name,
        layers_to_extract_from,
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        embedding_to_extract_from: str = None,
    ):
        super(FeatureExtractor, self).__init__()

        self.backbone = get_backbone(backbone_name)

        self.input_shape = input_shape
        self.layers_to_extract_from = layers_to_extract_from

        self.patch_maker = PatchMaker(patchsize, patchstride)

        all_layers_to_extract_from = list(layers_to_extract_from)

        if embedding_to_extract_from:
            all_layers_to_extract_from += [embedding_to_extract_from]
            self.embedding_to_extract_from = embedding_to_extract_from

        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, all_layers_to_extract_from, torch.device("cpu")
        )
        feature_aggregator.eval()
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)[
            : len(layers_to_extract_from)
        ]

        preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension)

        preadapt_aggregator = Aggregator(target_dim=target_embed_dimension)

        self.forward_modules = torch.nn.ModuleDict({})
        self.forward_modules["feature_aggregator"] = feature_aggregator
        self.forward_modules["preprocessing"] = preprocessing
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.eval()
        self.feature_map_shape = self._compute_feature_map_shape()
        self.target_embed_dimension = target_embed_dimension

    @torch.no_grad()
    def forward(
        self, images, detach=True, provide_patch_shapes=False, return_embeddings=False
    ) -> torch.Tensor:
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return features.detach().cpu()
            return features

        features = self.forward_modules["feature_aggregator"](images)

        if return_embeddings:
            embeddings = features[self.embedding_to_extract_from]

        features = [
            features[layer] for layer in self.layers_to_extract_from
        ]  # [(b, c, #p, #p), ..]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )  # (batch, #patch, #patch, channel, kernel, kernel))
            _features = _features.permute(
                0, -3, -2, -1, 1, 2
            )  # (batch, channel, kernel, kernel, #patch, #patch)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(
                0, -2, -1, 1, 2, 3
            )  # (batch, #patch, #patch, channel, kernel, kernel)
            _features = _features.reshape(
                len(_features), -1, *_features.shape[-3:]
            )  # (batch, #patch*#patch, channel, kernel, kernel)
            features[i] = _features
        features = [
            x.reshape(-1, *x.shape[-3:]) for x in features
        ]  # list of (#total, channel, kernel, kernel)

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](
            features
        )  # (#total, #layers, designated_channel*kernel*kernel -> pretrain_embed_dimension)
        features = self.forward_modules["preadapt_aggregator"](
            features
        )  # (#total, designated_channel -> target_embed_dimension) (layers are averaged)

        if provide_patch_shapes:
            return _detach(features), patch_shapes

        if return_embeddings:
            return _detach(features), _detach(embeddings)

        return _detach(features)

    def _compute_feature_map_shape(self):
        self.eval()
        _device = next(self.parameters()).device
        _input = torch.ones([1] + list(self.input_shape)).to(_device)
        dummy_feas, feature_map_shapes = self(_input, provide_patch_shapes=True)
        return feature_map_shapes[0] + [dummy_feas.shape[-1]]

    def get_feature_map_shape(self):
        return self.feature_map_shape