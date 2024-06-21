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

from model7_shift_scale import UNETv13
import torch.nn.functional as func

from main import ONEPW_Dataset2

# ################################
# file_log = open("/CODIGOS_TESIS/T2/trained_models/DDPM_model/v6_TT_50epoch/log_sampling1.txt", "w")
# file_log.close()
# ################################
# def write_to_file(input): 
#     #file_log = open("/CODIGOS_TESIS/T2/trained_models/DDPM_model/v6_TT_50epoch/log_sampling1.txt", "a")
#     with open("/CODIGOS_TESIS/T2/trained_models/DDPM_model/v6_TT_50epoch/log_sampling1.txt", "a") as textfile: 
#         textfile.write(str(input) + "\n") 
#     textfile.close()

'''
DATASET
'''
#creating our own Dataset
#esta clase va a heredar de la clase Dataset de Pytorch
class ONEPW_Dataset(Dataset):
    def __init__(self, data, onepw_img):
        '''
        data - train data path
        enh_img - train enhanced images path
        '''
        self.train_data = data
        self.train_onepw_img = onepw_img

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
        rf_image = torch.Tensor(rf_image)
        rf_image = rf_image.permute(2, 0, 1)

        onepw_image_name = os.path.join(self.train_onepw_img, self.onepw_images[idx])
        onepw_img = np.load(onepw_image_name)
        onepw_img = torch.Tensor(onepw_img)
        onepw_img = onepw_img.unsqueeze(0)
        new_min = -1
        new_max = 1
        onepw_img = onepw_img * (new_max - new_min) + new_min

        return rf_image, onepw_img
    
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        # transforms.Lambda(lambda t: (t * 60) - 60.),
        transforms.Lambda(lambda t: t.numpy())
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image), cmap='gray', extent=[-20,20,50,0])
    # plt.clim(-60,0)


def create_gaussian_diffusion(
        *,
        steps=100,
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

def show_reverse_process(intermediate):
    """ Shows a list of tensors from the sampling process """
    num_intermediate = len(intermediate)
    plt.figure(figsize=(15,2))
    plt.axis('off')
    for id, y_gen in enumerate(intermediate):
        plt.subplot(1, num_intermediate, id+1)
        show_tensor_image(y_gen)
    plt.show()

def main():

    diffusion = create_gaussian_diffusion()

    # TRAIN_PATH = '/mnt/nfs/efernandez/datasets/dataRF/RF_train'
    # TRAIN_ENH_PATH= '/mnt/nfs/efernandez/datasets/dataENH/ENH_train'
    # TRAIN_ONEPW_PATH= '/mnt/nfs/efernandez/datasets/dataONEPW/ONEPW_train'

    TRAIN_PATH = '/TESIS/DATOS_1/rf_train'
    TRAIN_ENH_PATH= '/TESIS/DATOS_1/enh_train'
    TRAIN_ONEPW_PATH= '/TESIS/DATOS_TESIS2/onepw_train'

    # TEST_PATH = '/mnt/nfs/efernandez/datasets/dataRF/RF_test'
    # TEST_ENH_PATH= '/mnt/nfs/efernandez/datasets/dataENH/ENH_test'
    # TEST_ONEPW_PATH= '/mnt/nfs/efernandez/datasets/dataONEPW/ONEPW_test'

    TEST_PATH = '/TESIS/DATOS_1/rf_test'
    TEST_ENH_PATH='/TESIS/DATOS_1/enh_test'
    TEST_ONEPW_PATH='/TESIS/DATOS_TESIS2/onepw_test'

    TRAIN_75PW_16m_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_16neg/'
    TRAIN_75PW_8m_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_8neg/'
    TRAIN_75PW_0_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_zero/'
    TRAIN_75PW_8p_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_8pos/'
    TRAIN_75PW_16p_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_16pos/'

    TRAIN_75PW_PATH= '/mnt/nfs/efernandez/datasets/data75PW/75PW_train'

    BATCH_SIZE = 1

    #data = gd.CustomDataset(TRAIN_PATH, TRAIN_ONEPW_PATH, transform=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    print(device)

    # train_dataset = ONEPW_Dataset(TRAIN_PATH, TRAIN_ONEPW_PATH)
    train_dataset = ONEPW_Dataset2(TRAIN_75PW_16m_PATH, TRAIN_75PW_8m_PATH, TRAIN_75PW_0_PATH, TRAIN_75PW_8p_PATH, TRAIN_75PW_16p_PATH, TRAIN_75PW_PATH)

    # test_dataset = ONEPW_Dataset(TEST_PATH, TEST_ONEPW_PATH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # save_dir = '/CODIGOS_TESIS/T2/generated_samples/DDPM_model/v6_TT_300epoch_gen_samples'
    save_dir = '/CODIGOS_TESIS/T2/generated_samples/DDPM_model/v6_TT_100steps_75pw_samples'
    # save_dir = '/CODIGOS_TESIS/T2/generated_samples/UNET_Difusiva/v1_50epoch_gen_samples'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_dir='/CODIGOS_TESIS/T2/trained_models/DDPM_model/v6_TT_100steps'
    # model_dir='/CODIGOS_TESIS/T2/trained_models/UNET_Difusiva/'
    # model_dir = '/CODIGOS_TESIS/T2/trained_models/DDPM_model/v6_TT_50epoch'
    training_epochs = 380#10
    model13A = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
    model13A.load_state_dict(torch.load(f"{model_dir}/model_{training_epochs}.pth", map_location=device))

    #print("Num params: ", sum(p.numel() for p in model13A.parameters()))

    mse_loss=[]
    num_samples = 0

    # write_to_file("hola")
    batch_number=0

    torch.manual_seed(2809)
    for i in range(1,11):
        x, y = next(iter(train_dataloader))
        x = x.to(device)
        y = y.to(device)

        # intermediate = []
        # for step in diffusion.p_sample_loop_progressive(model13A, y.shape, x, progress=True, clip_denoised=True):
        #     intermediate.append(step['sample'].cpu().detach())

        # intermediates=intermediate[::10]
        # intermediates.append(intermediate[-1])
        # show_reverse_process(intermediates)


        generated_samples = diffusion.p_sample_loop(model13A, y.shape, x, progress=True, clip_denoised=True)
        # generated_samples=model13A(x)

    # # loss is mean squared error between the predicted and true noise
    # mse_loss.append(func.mse_loss(generated_image, y))

    # np.save(save_dir+f"/test_mse_loss.npy", np.array(mse_loss))
    # print(f'Test_mse: {sum(mse_loss)/len(mse_loss)}')
        print(generated_samples.shape)

        for id, sample in enumerate(generated_samples):
            num_samples=num_samples+1
            # # sample = sample[i, :, :, :]
            # plt.figure(figsize=(9, 3))
            # # plt.subplot(1, 2, 1)
            # show_tensor_image(intermediate[-1].cpu().detach())
            # plt.colorbar()
            # plt.title('ENH')
            # plt.show()
            # sample = sample.cpu().numpy()
            name = (train_dataset.images[batch_number * BATCH_SIZE + id])
            print(name)
            print(sample.shape)
            np.save(save_dir+f"/{name}", sample.cpu().detach().numpy())

        batch_number += 1

        # if num_samples==BATCH_SIZE or num_samples%10==0:
        #     # write_to_file(num_samples)
        
        # for i in range(BATCH_SIZE):
        #     y = y[i, :, :, :]
        #     plt.figure(figsize=(9, 3))
        #     # plt.subplot(1, 2, 1)
        #     show_tensor_image(y.cpu().detach())
        #     plt.colorbar()
        #     plt.title('1PW')
        #     plt.show()


if __name__ == '__main__':
    main()
