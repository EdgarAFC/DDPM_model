import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os

# from plotting_utils import *
import torch
import numpy as np

from torchvision import transforms as T
import guided_diffusion_v3 as gd
from torchvision import transforms
import PIL
from PIL import Image

from model7 import UNETv13
import torch.nn.functional as func

'''
DATASET
'''
#creating our own Dataset
#esta clase va a heredar de la clase Dataset de Pytorch
class ONEPW_Dataset(Dataset):
    def __init__(self, data, onepw_img=None, img_transforms=None, onepw_img_transforms=None):
        '''
        data - train data path
        enh_img - train enhanced images path
        '''
        self.train_data = data
        self.train_onepw_img = onepw_img
        self.img_transforms = img_transforms
        self.onepw_img_transforms = onepw_img_transforms

        self.images = sorted(os.listdir(self.train_data))
        self.onepw_images = sorted(os.listdir(self.train_onepw_img))
  
    #regresar la longitud de la lista, cuantos elementos hay en el dataset
    def __len__(self):
        if self.onepw_images is not None:
            assert len(self.images) == len(self.onepw_images), 'not the same number of images ans enh_images'
        return len(self.images)

    def __getitem__(self, idx):
        rf_image_name = os.path.join(self.train_data, self.images[idx])
        rf_image = np.load(rf_image_name)
        trans = T.ToTensor()

        if self.img_transforms is not None:
            rf_image = Image.fromarray(rf_image)
            rf_image = self.img_transforms(rf_image)
        else:
            rf_image = trans(np.float32(rf_image))


        if self.train_onepw_img is not None:
            onepw_image_name = os.path.join(self.train_onepw_img, self.onepw_images[idx])
            onepw_img = np.load(onepw_image_name)
            if self.onepw_img_transforms is not None:
                onepw_img = Image.fromarray(onepw_img)
                onepw_img = self.onepw_img_transforms(onepw_img)
            else:
                onepw_img = trans(np.float32(onepw_img))
                new_min = -1
                new_max = 1
                onepw_img = onepw_img * (new_max - new_min) + new_min
        else:
            return rf_image

        return rf_image, onepw_img


def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return gd.SpacedDiffusion(
        use_timesteps=gd.space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

############

diffusion = create_gaussian_diffusion()

TRAIN_PATH = '/mnt/nfs/efernandez/datasets/dataRF/RF_train'
TRAIN_ENH_PATH= '/mnt/nfs/efernandez/datasets/dataENH/ENH_train'
TRAIN_ONEPW_PATH= '/mnt/nfs/efernandez/datasets/dataONEPW/ONEPW_train'

TEST_PATH = '/mnt/nfs/efernandez/datasets/dataRF/RF_test'
TEST_ENH_PATH= '/mnt/nfs/efernandez/datasets/dataENH/ENH_test'
TEST_ONEPW_PATH= '/mnt/nfs/efernandez/datasets/dataONEPW/ONEPW_test'

BATCH_SIZE = 4

#data = gd.CustomDataset(TRAIN_PATH, TRAIN_ONEPW_PATH, transform=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print(device)

train_dataset = ONEPW_Dataset(TRAIN_PATH, TRAIN_ONEPW_PATH)
test_dataset = ONEPW_Dataset(TEST_PATH, TEST_ONEPW_PATH)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

save_dir = '/mnt/nfs/efernandez/trained_models/DDPM_model/v6_TT_50epoch'
training_epochs = 50#10
model13A = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
model13A.load_state_dict(torch.load(f"{save_dir}/model_{training_epochs}.pth", map_location=device))

#print("Num params: ", sum(p.numel() for p in model13A.parameters()))

mse_loss=[]

for x, y in test_dataloader:
    x = x.to(device)
    y = y.to(device)

    generated_image = diffusion.p_sample_loop(model13A, y.shape, x, progress=False, clip_denoised=True)

    # loss is mean squared error between the predicted and true noise
    mse_loss.append(func.mse_loss(generated_image, y))

np.save(save_dir+f"/test_mse_loss.npy", np.array(mse_loss))
print(f'Test_mse: {sum(mse_loss)/len(mse_loss)}')
