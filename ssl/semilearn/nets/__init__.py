# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys

from .resnet import resnet50, resnet34, resnet18
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2
# from .vit import vit_base_patch16_224, vit_small_patch16_224, vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96
# from .bert import bert_base_cased, bert_base_uncased
# from .wave2vecv2 import wave2vecv2_base
# from .hubert import hubert_base
from .cifar10_models import resnet32
from .mobilenetv2 import mobilenet_v2
from .vit_pytorch import vit_b_16
# from torchvision.models.vision_transformer import vit_b_16
from .resnet_big import SupConResNet34 as supconresnet34
from .gbcnet import GbcNet
from .radformer.radformer import radformer
from .deit import deit_base_patch16_224
<<<<<<< HEAD

sys.path.append('/home/ankita/scratch/MSA_Medical')
from activethief.inception import inception_v3
# from .visiontransformer import vit_b_16
=======
from .visiontransformer import vit_b_16
>>>>>>> ba1060430d61521041e28b47a409dc50ce80da3d
