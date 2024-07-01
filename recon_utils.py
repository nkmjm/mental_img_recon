import os
import pickle
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output, display
import recon_func


def convert_featname(names, cvt_to='directory'):
    new_names = list()
    for name in names:
        if cvt_to == 'directory':
            name = (name.replace('[', '_layer')).replace(']', '')
        elif cvt_to == 'model':
            name = (name.replace('_layer', '[')) + ']'
        else:
            print(
                '(no-processed name) ''cvt_to'' should be set to ''directory'' or ''model''. ')
        new_names.append(name)

    return new_names


def get_target_image(targetID, targetimpath):
    with open(targetimpath, 'rb') as f:
        dt = pickle.load(f)

    target_images = dt['target_images']
    target_labels = dt['target_labels']
    target_positions = dt['target_positions']
    imsize = dt['imsize']
    t_label = target_labels[targetID]
    t_pos = target_positions[t_label]
    t_im = target_images[t_pos[1]:t_pos[1] +
                         imsize[1], t_pos[0]:t_pos[0]+imsize[0]]

    return t_im, t_label


def update_subplot(row, col, new_image, title, axes):
    if len(axes.shape) == 1:
        ax = axes[col]
    else:
        ax = axes[row, col]
    ax.imshow(new_image)
    ax.set_title(title)
    ax.axis('off')


def show_reconstruction_info(subject, reconMethod, feat_set, targetimname, CLIPmodelWeight_, CLIPcoef_, VGGlayerWeight_, loss_VGG=None, loss_CLIP=None, time_step_text=None):
    print(f'Subject: {subject}, feat_set: {feat_set}, Image: {targetimname}')
    print(f'Reconstruction method: {reconMethod}')
    print(f'CLIPmodelWeight: {CLIPmodelWeight_}')
    print(f'CLIP coef: {CLIPcoef_}')
    print(f'VGGs layer weight: {VGGlayerWeight_}')
    print('-------------------------------------------')
    if loss_VGG is not None and loss_CLIP is not None:
        print(f'Loss VGG: {loss_VGG}, Loss CLIP: {loss_CLIP}')
    if time_step_text is not None:
        print(time_step_text)


def display_reconstruction_progress(fig, axes, row, col, new_image, title, subject, reconMethod, feat_set, targetimname, CLIPmodelWeight_, CLIPcoef_, VGGlayerWeight_, loss_VGG=None, loss_CLIP=None, time_step_text=None):
    # prepare new subplot
    update_subplot(row, col, new_image, title, axes)
    plt.tight_layout()
    # delete previous output
    clear_output(wait=True)
    # display new output
    show_reconstruction_info(
        subject, reconMethod, feat_set, targetimname,
        CLIPmodelWeight_, CLIPcoef_, VGGlayerWeight_, loss_VGG, loss_CLIP, time_step_text)
    display(fig)


def start_reconstruction(subject, targetID_list, reconMethod, dt_cfg, prm_demo, CLIPmodel_, VGGmodel_, CLIPmodelWeight_, VQGANmodel1024, DEVICE):
    torch.cuda.empty_cache()

    # Set parameters
    meanFeatureDir = dt_cfg['file_path']['mean_feat_dir']  
    targetimpath = prm_demo['dt_targetimages_path']
    CLIP_modelNames = dt_cfg["models"]["CLIP"]["modelnames"]
    CLIP_usedLayer = dt_cfg["models"]["CLIP"]["used_layer"]
    CLIPmodelWeight_ = dt_cfg["models"]["CLIP"]["modelcoefs"]

    # Set plot
    number_of_targets = len(targetID_list)
    fig, axes = plt.subplots(number_of_targets, 2, figsize=(5.0, 2.5 * number_of_targets))

    # Show all target images
    for i, targetID in enumerate(targetID_list):
        targetImg_, targetimname = get_target_image(targetID, targetimpath)
        update_subplot(i, 0, targetImg_, f'Target Image', axes)
    plt.tight_layout()
    display(fig)

    for i, targetID in enumerate(targetID_list):
        targetImg_, targetimname = get_target_image(targetID, targetimpath)
        
        # Load decfeat
        used_layers_VGG__in = dt_cfg["recon_feat_layers"]["all"]["VGG19"]
        used_layers_VGG = convert_featname(used_layers_VGG__in, cvt_to='directory')

        # VGG
        list_path_vgg = list()
        for t_layername in used_layers_VGG:
            path_decfeat = prm_demo['decfearture_path']
            path_decfeat = path_decfeat.replace('__subjectname__', subject)
            path_decfeat = path_decfeat.replace('__modelname__', 'VGG19')
            path_decfeat = path_decfeat.replace('__layername__', t_layername)
            path_decfeat = path_decfeat.replace('__targetimname__', targetimname)
            list_path_vgg.append(path_decfeat)

        # CLIP
        list_path_clip = list()
        for t_modelname in CLIP_modelNames:
            path_decfeat = prm_demo['decfearture_path']
            path_decfeat = path_decfeat.replace('__subjectname__', subject)
            path_decfeat = path_decfeat.replace('__modelname__', t_modelname)
            path_decfeat = path_decfeat.replace('__layername__', CLIP_usedLayer)
            path_decfeat = path_decfeat.replace('__targetimname__', targetimname)
            list_path_clip.append(path_decfeat)

        # Load CLIP & VGG19 features ---------------------------------------------

        # targetCLIPfeature_: 
        # Prepare target CLIP feature (decoded CLIP feature)
        targetCLIPfeature_ = list()
        for mi in range(len(CLIPmodel_)):
            with open(list_path_clip[mi], 'rb') as f:
                dt = pickle.load(f)
            x = dt[0].astype('float32')
            targetCLIPfeature_.append(torch.tensor(x, dtype=torch.float32).to(DEVICE).unsqueeze(0))
            del dt, x

        # Set meanCLIPfeature_:
        # Prepare the mean CLIP feature vector, which is used in the normalization (i.e., centering) process.
        meanCLIPfeature_ = list()
        for mi in range(len(CLIPmodel_)):
            x = scipy.io.loadmat(os.path.join(meanFeatureDir, CLIP_modelNames[mi], CLIP_usedLayer, 'meanFeature_.mat'))
            meanCLIPfeature_.append(torch.tensor(x['mu'], dtype=torch.float32).to(DEVICE))
        del x

        # targetVGGfeature_ (decoded VGG feature):
        targetVGGfeature_ = list()
        for li in range(len(used_layers_VGG)):
            with open(list_path_vgg[li], 'rb') as f:
                dt = pickle.load(f)
            x = dt[0].astype('float32')
            targetVGGfeature_.append(torch.tensor(x, dtype=torch.float32).to(DEVICE).unsqueeze(0))
            del dt, x

        meanVGGfeature_ = list()
        for li in range(len(used_layers_VGG)):
            x = scipy.io.loadmat(os.path.join(meanFeatureDir, 'VGG19', used_layers_VGG[li], 'meanFeature_.mat'))
            meanVGGfeature_.append(torch.tensor(x['mu'], dtype=torch.float32).to(DEVICE))
            del x

        ### Main: Reconstruction -------------------------------------------------
        VGGlayerWeight_ = np.ones(len(used_layers_VGG))
        VGGlayerWeight_ = VGGlayerWeight_/VGGlayerWeight_.sum()

        # Set the initial image
        initialImage_PIL_ = Image.open('./ref_images/uniformGray.tiff')

        # Set parameters
        CLIPcoef_ = dt_cfg['recon_params'][reconMethod]['clip_coef']
        feat_set = dt_cfg['recon_params'][reconMethod]['feat_set']
        disp_every = dt_cfg['recon_params'][reconMethod]['display_every']
        numReps = dt_cfg['recon_params'][reconMethod]['numReps']
        similarity = dt_cfg['recon_params'][reconMethod]['similarity']
        
        if reconMethod == 'Langevin' or reconMethod == 'original':
            lr_gamma = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_gamma']
            lr_a = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_a']
            lr_b = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_b']
            T_langevin = dt_cfg['recon_params'][reconMethod]['Langevin']['T']
        
        # Reconstruction
        reconf = recon_func.imageRecon(
            targetVGGfeature_, meanVGGfeature_, VGGlayerWeight_, VGGmodel_, used_layers_VGG__in,
            targetCLIPfeature_, meanCLIPfeature_, CLIPmodelWeight_, CLIPmodel_,
            VQGANmodel1024, initialImage_PIL_, initInputType='PIL',
            similarity=similarity, disp_every=disp_every, numReps=numReps, CLIPcoef=CLIPcoef_, DEVICE=DEVICE
        )

        if reconMethod == 'withoutLangevin':
            generator = reconf.withoutLangevin()
            for recImg, time_step, loss_VGG, loss_CLIP, currentLatentVec in generator:
                display_reconstruction_progress(
                    fig, axes, i, 1, recImg, f'Reconstructed Image: {time_step} / {numReps}',
                    subject, reconMethod, feat_set, targetimname,
                    CLIPmodelWeight_, CLIPcoef_, VGGlayerWeight_,
                    loss_VGG, loss_CLIP, time_step
                )

        elif reconMethod == 'Langevin':
            generator = reconf.Langevin(lr_gamma=lr_gamma, lr_a=lr_a, lr_b=lr_b, T=T_langevin)
            for recImg, time_step, loss_VGG, loss_CLIP, currentLatentVec in generator:
                display_reconstruction_progress(
                    fig, axes, i, 1, recImg, f'Reconstructed Image: {time_step} / {numReps}',
                    subject, reconMethod, feat_set, targetimname,
                    CLIPmodelWeight_, CLIPcoef_, VGGlayerWeight_,
                    loss_VGG, loss_CLIP, f'Time step: {time_step} / {numReps}'
                )

        elif reconMethod == 'original':
            # set parameters
            numReps_withoutLangevin = 1000 # (default) 1000
            numReps_Langevin = 500 # (default) 500

            print('Reconstruction without Langevin:')
            generator = reconf.withoutLangevin(numReps=numReps_withoutLangevin, returnVec=True)
            currentLatentVec = None
            for recImg, time_step, loss_VGG, loss_CLIP, currentLatentVec in generator:
                display_reconstruction_progress(
                    fig, axes, i, 1, recImg, 
                    f'Reconstructed Image: {time_step} / {numReps_Langevin + numReps_withoutLangevin}',
                    subject, reconMethod, feat_set, targetimname,
                    CLIPmodelWeight_, CLIPcoef_, VGGlayerWeight_,
                    loss_VGG, loss_CLIP,
                    f'Time step: {time_step} / {numReps_Langevin + numReps_withoutLangevin}'
                )

            print('Reconstruction using Langevin:')
            generator = reconf.Langevin(initInput=currentLatentVec, initInputType='latentVector', numReps=numReps_Langevin)
            for recImg, time_step, loss_VGG, loss_CLIP, currentLatentVec in generator:
                display_reconstruction_progress(
                    fig, axes, i, 1, recImg, 
                    f'Reconstructed Image: {time_step + numReps_withoutLangevin} / {numReps_Langevin + numReps_withoutLangevin}',
                    subject, reconMethod, feat_set, targetimname,
                    CLIPmodelWeight_, CLIPcoef_, VGGlayerWeight_,
                    loss_VGG, loss_CLIP,
                    f'Time step: {time_step + numReps_withoutLangevin} / {numReps_Langevin + numReps_withoutLangevin}'
                )

        display_reconstruction_progress(
            fig, axes, i, 1, recImg, 
            f'Reconstructed Image',
            subject, reconMethod, feat_set, targetimname,
            CLIPmodelWeight_, CLIPcoef_, VGGlayerWeight_,
            loss_VGG, loss_CLIP
        )

        torch.cuda.empty_cache()

    clear_output(wait=True)
    print('Reconstruction is completed.')
    print(f'Subject: {subject}, method: {reconMethod}')