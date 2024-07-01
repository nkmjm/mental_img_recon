
import yaml
import torch
import taming
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import clip



###
### For model loading


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def load_VGG_model(DEVICE, pretrained=True):
    
    from torchvision import transforms
    import torchvision.models
    
    preprocessForVGG = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                                         ])
    VGGmodel_ = torchvision.models.vgg19(pretrained=pretrained)
    VGGmodel_.eval()
    VGGmodel_.to(DEVICE)
    
    return VGGmodel_, preprocessForVGG



def load_CLIP_model(model_names_CLIP, DEVICE): # @@@

    # Load CLIP models to be used.
    # We can use the below five CLIP models.
    # ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    # Here, RN50 and RN101 are used with equal weights, 
    # but you can combine all five models with arbitrary weights if your available GPU memory allosw it.
    
    nameOfCLIPmodelInTorch = list()
    for model_name in model_names_CLIP:
        nameOfCLIPmodelInTorch.append(model_name)

    nameOfSubdirForCLIPfeature = list()
    for mi in range(len(model_names_CLIP)):
        temporal_NameOfSubdir = 'CLIP_' + nameOfCLIPmodelInTorch[mi]
        nameOfSubdirForCLIPfeature.append(temporal_NameOfSubdir.replace('/','_'))

    print('nameOfCLIPmodelInTorch:')
    print(nameOfCLIPmodelInTorch)
    print('nameOfSubdirForCLIPfeature:')
    print(nameOfSubdirForCLIPfeature)
    CLIPmodel_ = list()
    for mi in range(len(model_names_CLIP)):
        temporal_model, temporal_preprpcess = clip.load(nameOfCLIPmodelInTorch[mi],jit=False, device = DEVICE)
        CLIPmodel_.append(temporal_model)
        CLIPmodel_[mi].eval()
    #torch.cuda.empty_cache()

    return CLIPmodel_, nameOfSubdirForCLIPfeature

