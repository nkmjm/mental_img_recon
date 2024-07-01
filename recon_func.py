import sys
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import copy
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm

device_default = torch.device("cuda:0")


###
# Preprocessing input images

# Converting VQGAN output into CLIP input format
def convertVQGANoutputIntoCLIPinput(VQGANoutput, imageSize=[224, 224]):

    VQGANoutput = (VQGANoutput+1.0)*0.5
    if VQGANoutput.shape[2] == imageSize[0] and VQGANoutput.shape[3] == imageSize[1]:
        preprocessBeforeCLIP = transforms.Compose([
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])])
    else:
        preprocessBeforeCLIP = transforms.Compose([
            transforms.Resize((imageSize[0], imageSize[1])),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])])
    inputForCLIP = preprocessBeforeCLIP(VQGANoutput)

    return inputForCLIP

# Converting VQGAN output into VGG input format
def convertVQGANoutputIntoVGGinput(VQGANoutput):

    VQGANoutput = (VQGANoutput+1.0)*0.5
    if VQGANoutput.shape[2] == 224 and VQGANoutput.shape[3] == 224:
        preprocessBeforeVGG = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        preprocessBeforeVGG = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    inputForVGG = preprocessBeforeVGG(VQGANoutput)

    return inputForVGG

# Preprocessing for CLIP input image
def createCrops(img, num_crops=32, DEVICE=device_default):

    size1 = img.shape[2]
    size2 = img.shape[3]
    augTransform = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(30, (.2, .2), fill=0)
    ).to(DEVICE)
    noise_factor = 0.22

    p = size1//2
    # 1 x 3 x 672 x 672 (adding 112*2 on all sides to 448x448)
    img = torch.nn.functional.pad(img, (p, p, p, p), mode='constant', value=0)
    img = augTransform(img)  # RandomHorizontalFlip and RandomAffine
    crop_set = []
    for ch in range(num_crops):
        gap1 = int(torch.normal(1.2, .3, ()).clip(.43, 1.9) * size1)
        offsetx = torch.randint(0, int(size1*2-gap1), ())
        offsety = torch.randint(0, int(size1*2-gap1), ())
        crop = img[:, :, offsetx:offsetx+gap1, offsety:offsety+gap1]
        crop = torch.nn.functional.interpolate(
            crop, (size1, size2), mode='bilinear', align_corners=True)
        crop_set.append(crop)
    img_crops = torch.cat(crop_set, 0)  # 30 x 3 x 224 x 224

    randnormal = torch.randn_like(img_crops, requires_grad=False)

    randstotal = torch.rand((img_crops.shape[0], 1, 1, 1)).to(DEVICE)  # 32

    img_crops = img_crops + noise_factor*randstotal*randnormal

    return img_crops


###
# Get features from preproc input images
# CLIP : using CLIPmodel[index_CLIPmodel].encode_image
# VGG
def extract_VGG_features(model, input, target_layers, prehook_dict={}):

    model_ = copy.deepcopy(model)

    def hook(module, input, output):
        outputs_.append(output.clone())

    def hook_pre(module, input):
        for v in val_list:
            outputs_.append(input[0][v].clone())

    for layer in target_layers:
        t_layername = layer.split('[')[0]
        t_layerno = int(layer.split('[')[1].replace(']', ''))
        target_layer = getattr(model_, t_layername)
        if layer not in prehook_dict:
            target_layer[t_layerno].register_forward_hook(hook)
        else:
            val_list = prehook_dict[layer]
            target_layer[t_layerno].register_forward_pre_hook(hook_pre)
        del t_layername, t_layerno

    outputs_ = []
    model_(input)
    del model_

    return outputs_


###
# Compute Loss
def compute_loss_CLIP(CLIPmodel, CLIPmodelWeight, input1, input1_type, input2, input2_type, meanCLIPfeature, cosSimilarity, similarity='corr', DEVICE=device_default):

    for index_CLIPmodel in range(len(CLIPmodel)):

        # Get x1, x2
        # for input1
        if input1_type == 'img':
            CLIPfeature1 = CLIPmodel[index_CLIPmodel].encode_image(
                createCrops(input1, DEVICE=DEVICE))
            x1 = CLIPfeature1.reshape(CLIPfeature1.shape[0], -1)
        elif input1_type == 'feat':
            x1 = input1[index_CLIPmodel]
        # for input2
        if input2_type == 'img':
            CLIPfeature2 = CLIPmodel[index_CLIPmodel].encode_image(
                createCrops(input2, DEVICE=DEVICE))
            x2 = CLIPfeature2.reshape(CLIPfeature2.shape[0], -1)
        elif input2_type == 'feat':
            x2 = input2[index_CLIPmodel]

        # Subtract the mean feature vector
        x1 = x1-meanCLIPfeature[index_CLIPmodel].reshape(1, -1)
        x2 = x2-meanCLIPfeature[index_CLIPmodel].reshape(1, -1)

        # Compute similarity
        if similarity == 'corr':
            loss_thisLayer = -cosSimilarity(x1-x1.mean(dim=1, keepdim=True),
                                            x2-x2.mean(dim=1, keepdim=True)).mean()
        elif similarity == 'cosine':
            loss_thisLayer = -cosSimilarity(x1, x2).mean()
        elif similarity == 'MSE':
            loss_thisLayer = ((x1-x2)**2).mean()
        else:
            print('Error: Similarity metric should be corr, cosine, or MSE.')
            sys.exit(1)

        if index_CLIPmodel == 0:
            loss_CLIP = loss_thisLayer*CLIPmodelWeight[index_CLIPmodel]
        else:
            loss_CLIP = loss_thisLayer * \
                CLIPmodelWeight[index_CLIPmodel]+loss_CLIP

    return loss_CLIP


def compute_loss_VGG(VGGmodel, usedLayer, layerWeight, input1, input1_type, input2, input2_type, meanVGGfeature, cosSimilarity, similarity='corr'):

    if input1_type == 'img':
        VGGfeature_allLayer1 = extract_VGG_features(
            VGGmodel, input1, usedLayer)  # get_cnn_features
    if input2_type == 'img':
        VGGfeature_allLayer2 = extract_VGG_features(
            VGGmodel, input2, usedLayer)
    for index_usedLayer in range(len(usedLayer)):
        # for input1
        if input1_type == 'img':
            x1 = VGGfeature_allLayer1[index_usedLayer].reshape(
                VGGfeature_allLayer1[index_usedLayer].shape[0], -1)
        elif input1_type == 'feat':
            x1 = input1[index_usedLayer].reshape(1, -1)
        # for input2
        if input2_type == 'img':
            x2 = VGGfeature_allLayer2[index_usedLayer].reshape(
                VGGfeature_allLayer2[index_usedLayer].shape[0], -1)
        elif input2_type == 'feat':
            x2 = input2[index_usedLayer].reshape(1, -1)

        # Subtract the mean feature vector
        x1 = x1-meanVGGfeature[index_usedLayer].reshape(1, -1)
        x2 = x2-meanVGGfeature[index_usedLayer].reshape(1, -1)

        if similarity == 'corr':
            loss_thisLayer = -cosSimilarity(x1-x1.mean(dim=1, keepdim=True),
                                            x2-x2.mean(dim=1, keepdim=True)).mean()
        elif similarity == 'cosine':
            loss_thisLayer = -cosSimilarity(x1, x2).mean()
        elif similarity == 'MSE':
            loss_thisLayer = ((x1-x2)**2).mean()
        else:
            print('Error: Similarity metric should be corr, cosine, or MSE.')
            sys.exit(1)
        if index_usedLayer == 0:
            loss_VGG = loss_thisLayer*layerWeight[index_usedLayer]
        else:
            loss_VGG = loss_thisLayer*layerWeight[index_usedLayer] + loss_VGG

        del x1, x2

    return loss_VGG


###
# Utils in imageRecon

def set_initInput(initInput, initInputType, VQGANmodel, DEVICE=device_default):

    # The initial image given by the user is transformed into VQGAN's latent vector

    if initInputType == 'PIL':
        VQGANlatentSize = 14*1
        conversionFromPILintoTorchTensor = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize((VQGANlatentSize*16, VQGANlatentSize*16)),
            transforms.ToTensor()
        ])
        currentLatentVector = conversionFromPILintoTorchTensor(initInput)
        currentLatentVector = currentLatentVector.unsqueeze(0)
        currentLatentVector = currentLatentVector.to(DEVICE)
        currentLatentVector = (currentLatentVector*2)-1
        currentLatentVector = VQGANmodel.encode(currentLatentVector)
        currentLatentVector = currentLatentVector[0].detach(
        ).clone().to(DEVICE).requires_grad_()

    elif initInputType == 'latentVector':
        currentLatentVector = initInput

    return currentLatentVector


def set_optimizer(currentLatentVector, optimizer, lr):
    # Set the gradient method (AdamW or Adam).
    if optimizer == 'AdamW':
        op = optim.AdamW([currentLatentVector], lr=lr, weight_decay=0.1)
    elif optimizer == 'Adam':
        op = optim.Adam([currentLatentVector], lr=lr)
    elif optimizer == 'SGD':
        op = optim.SGD([currentLatentVector], lr=lr)
    else:
        print('Error: Optimizer must be Adam, AdamW, or SGD.')
        sys.exit(1)
    return op


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1, 2, 0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def get_recImg(VQGANmodel, currentLatentVector, DEVICE=device_default):

    latentVector_copied = currentLatentVector.detach().clone().to(DEVICE)
    with torch.no_grad():
        VQGANoutput = VQGANmodel.decode(latentVector_copied)
    VQGANoutput = VQGANoutput.clip(-1.0, 1.0)
    recImg = custom_to_pil(VQGANoutput[0, :, :, :])

    return recImg

###
# MAIN func.: image reconstruction

class imageRecon():

    def __init__(self,
                 targetVGGfeature,
                 meanVGGfeature,
                 layerWeight,
                 VGGmodel,
                 usedLayer,
                 targetCLIPfeature,
                 meanCLIPfeature,
                 CLIPmodelWeight,
                 CLIPmodel,
                 VQGANmodel,
                 initInput,
                 initInputType='PIL',
                 similarity='corr',
                 disp_every=100,
                 numReps=500,
                 CLIPcoef=0.25,
                 DEVICE=device_default
                 ):

        self.targetVGGfeature = targetVGGfeature
        self.meanVGGfeature = meanVGGfeature
        self.layerWeight = layerWeight
        self.VGGmodel = VGGmodel
        self.usedLayer = usedLayer
        self.targetCLIPfeature = targetCLIPfeature
        self.meanCLIPfeature = meanCLIPfeature
        self.CLIPmodelWeight = CLIPmodelWeight
        self.CLIPmodel = CLIPmodel
        self.VQGANmodel = VQGANmodel
        self.initInput = initInput
        self.initInputType = initInputType
        self.disp_every = disp_every
        self.numReps = numReps
        self.similarity = similarity
        self.CLIPcoef = CLIPcoef
        self.DEVICE = DEVICE

    def withoutLangevin(self, initInput=None, initInputType='PIL', optimizer='Adam', lr=0.5, T=1.0, numReps=None, returnVec=False):

        if numReps is None:
            numReps = self.numReps

        # The initial image given by the user is transformed into VQGAN's latent vector
        if initInput is None:
            initInput = self.initInput
            initInputType = self.initInputType

        currentLatentVector = set_initInput(
            initInput, initInputType, self.VQGANmodel, DEVICE=self.DEVICE)

        # Set the gradient method (AdamW or Adam).
        op = set_optimizer(currentLatentVector, optimizer, lr)

        # Parepare basic functions to be used later.
        cosSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        # Start iterative optimization.
        for index_rep in tqdm(range(numReps), ncols=100, desc="Progress rate"):  # range(numReps)

            # The latent vector is converted into the corresponding image.
            VQGANoutput = self.VQGANmodel.decode(currentLatentVector)
            VQGANoutput_VGG = convertVQGANoutputIntoVGGinput(VQGANoutput)
            VQGANoutput_CLIP = convertVQGANoutputIntoCLIPinput(VQGANoutput)

            VQGANoutput.requires_grad_()
            VQGANoutput_VGG.requires_grad_()
            VQGANoutput_CLIP.requires_grad_()

            # Compute the semantic similarity based on CLIP features
            loss_CLIP = compute_loss_CLIP(self.CLIPmodel, self.CLIPmodelWeight, VQGANoutput_CLIP, 'img',
                                          self.targetCLIPfeature, 'feat', self.meanCLIPfeature, cosSimilarity, DEVICE=self.DEVICE)

            # Compute the VGG similarity
            if len(self.usedLayer) == 0:
                loss_VGG = torch.tensor(0, dtype=torch.float32).to(self.DEVICE)
            else:
                loss_VGG = compute_loss_VGG(
                    self.VGGmodel, self.usedLayer, self.layerWeight,
                    VQGANoutput_VGG, 'img', self.targetVGGfeature,
                    'feat', self.meanVGGfeature, cosSimilarity)

            # Total loss
            loss = (loss_VGG + loss_CLIP * self.CLIPcoef)/T

            currentLatentVector.retain_grad()
            VQGANoutput.retain_grad()
            VQGANoutput_VGG.retain_grad()
            VQGANoutput_CLIP.retain_grad()

            # Set the grad of network to 0 as a preparation,
            # then do backpropagation
            self.VQGANmodel.zero_grad()
            for index_CLIPmodel in range(len(self.CLIPmodel)):
                self.CLIPmodel[index_CLIPmodel].zero_grad()
            self.VGGmodel.zero_grad()
            op.zero_grad()
            loss.backward(retain_graph=True)

            # Update
            op.step()
            loss_VGG_np = loss_VGG.detach().cpu().numpy()
            loss_CLIP_np = loss_CLIP.detach().cpu().numpy()

            # Plot the results.
            if (index_rep+1) % self.disp_every == 0:
                recImg = get_recImg(
                    self.VQGANmodel, currentLatentVector, DEVICE=self.DEVICE)
                yield recImg, index_rep+1, loss_VGG_np, loss_CLIP_np, None

        # Output the results and finish.
        recImg = get_recImg(
            self.VQGANmodel, currentLatentVector, DEVICE=self.DEVICE)

        if returnVec:
            yield recImg, index_rep+1, loss_VGG_np, loss_CLIP_np, currentLatentVector
        else:
            yield recImg, index_rep+1, loss_VGG_np, loss_CLIP_np, None

    # Define a function to perform image reconstruction
    # Reference for optimizer: https://tzmi.hatenablog.com/entry/2020/03/04/224258

    def Langevin(self, initInput=None, initInputType='PIL', lr_gamma=0.1, lr_a=1, lr_b=1, lrs=[], T=0.0001, numReps=None, returnVec=False):

        if numReps is None:
            numReps = self.numReps

        # Set lrs when lrs=[]
        if len(lrs) == 0:
            lrs = np.nan * np.ones(numReps)
            for t in range(numReps):
                lrs[t] = lr_a*((lr_b + t)**-lr_gamma)

        # The initial image given by the user is transformed into VQGAN's latent vector
        if initInput is None:
            initInput = self.initInput
            initInputType = self.initInputType

        currentLatentVector = set_initInput(
            initInput, initInputType, self.VQGANmodel, DEVICE=self.DEVICE)

        # Set the gradient method (SGD).
        size_currentLatentVector = np.prod(currentLatentVector.size())

        # Parepare basic functions to be used later.
        cosSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        # Start iterative optimization.
        t_lr = lrs[0]  # init state
        for index_rep in tqdm(range(numReps), ncols=100, desc="Progress rate"):  # range(numReps)

            # The latent vector is converted into the corresponding image.
            VQGANoutput = self.VQGANmodel.decode(currentLatentVector)
            VQGANoutput_VGG = convertVQGANoutputIntoVGGinput(VQGANoutput)
            VQGANoutput_CLIP = convertVQGANoutputIntoCLIPinput(VQGANoutput)

            VQGANoutput.requires_grad_()
            VQGANoutput_VGG.requires_grad_()
            VQGANoutput_CLIP.requires_grad_()

            # Compute the semantic similarity based on CLIP features
            loss_CLIP = compute_loss_CLIP(
                self.CLIPmodel, self.CLIPmodelWeight, VQGANoutput_CLIP, 'img',
                self.targetCLIPfeature, 'feat', self.meanCLIPfeature,
                cosSimilarity, DEVICE=self.DEVICE)

            # Compute the VGG similarity
            if len(self.usedLayer) == 0:
                loss_VGG = torch.tensor(0, dtype=torch.float32).to(self.DEVICE)
            else:
                loss_VGG = compute_loss_VGG(
                    self.VGGmodel, self.usedLayer, self.layerWeight,
                    VQGANoutput_VGG, 'img', self.targetVGGfeature, 'feat',
                    self.meanVGGfeature, cosSimilarity)

            # Total loss
            loss = (loss_VGG + loss_CLIP * self.CLIPcoef)

            currentLatentVector.retain_grad()
            VQGANoutput.retain_grad()
            VQGANoutput_VGG.retain_grad()
            VQGANoutput_CLIP.retain_grad()

            # Set the grad of network to 0 as a preparation,
            # then do backpropagation
            self.VQGANmodel.zero_grad()
            for index_CLIPmodel in range(len(self.CLIPmodel)):
                self.CLIPmodel[index_CLIPmodel].zero_grad()
            self.VGGmodel.zero_grad()
            loss.backward(retain_graph=True)

            # Update
            currentLatentVector = (
                currentLatentVector - currentLatentVector.grad * (t_lr/T)).detach().requires_grad_()

            # Langevin ------------------------------
            # Add gaussian noise into opt.state
            gauss_noise = np.random.normal(
                loc=0, scale=np.sqrt(t_lr), size=size_currentLatentVector)
            gauss_noise = torch.tensor(
                gauss_noise, dtype=torch.float32).to(self.DEVICE)
            gauss_noise = gauss_noise.reshape([1, 256, 14, 14])
            # Update learning rate (lr)
            if index_rep+1 != numReps:
                t_lr = lrs[index_rep+1]  # for the next step
            currentLatentVector = currentLatentVector + gauss_noise
            # Langevin ------------------------------

            loss_VGG_np = loss_VGG.detach().cpu().numpy()
            loss_CLIP_np = loss_CLIP.detach().cpu().numpy()

            # Plot the results.
            if (index_rep+1) % self.disp_every == 0:
                recImg = get_recImg(
                    self.VQGANmodel, currentLatentVector, DEVICE=self.DEVICE)
                yield recImg, index_rep+1, loss_VGG_np, loss_CLIP_np, None

        # Output the results and finish.
        recImg = get_recImg(
            self.VQGANmodel, currentLatentVector, DEVICE=self.DEVICE)

        if returnVec:
            yield recImg, index_rep+1, loss_VGG_np, loss_CLIP_np, currentLatentVector
        else:
            yield recImg, index_rep+1, loss_VGG_np, loss_CLIP_np, None
