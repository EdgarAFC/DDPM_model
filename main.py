import torch
import os
from pathlib import Path
# from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import guided_diffusion_v3 as gd
from datetime import datetime
import torch.nn.functional as func
from model7_shift_scale import UNETv13
# from model4 import UNETv10_5, UNETv10_5_2
import torch.nn as nn

from torchvision import transforms as T
import PIL
from PIL import Image

TRAIN_PATH = '/mnt/nfs/efernandez/datasets/dataRF/RF_train2'
TRAIN_ENH_PATH= '/mnt/nfs/efernandez/datasets/dataENH'
TRAIN_ONEPW_PATH= '/mnt/nfs/efernandez/datasets/dataONEPW/ONEPW_train'
TRAIN_75PW_PATH= '/mnt/nfs/efernandez/datasets/data75PW/75PW_train'


TRAIN_75PW_16m_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_16neg/'
TRAIN_75PW_8m_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_8neg/'
TRAIN_75PW_0_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_zero/'
TRAIN_75PW_8p_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_8pos/'
TRAIN_75PW_16p_PATH='/mnt/nfs/efernandez/datasets/dataMULTPW/MULTPW_train/angle_16pos/'
                

###############################
file_loss = open("/mnt/nfs/efernandez/log_files/DDPM_model/log_fv.txt", "w")
file_loss.close()
################################
def write_to_file(input): 
    with open("/mnt/nfs/efernandez/log_files/DDPM_model/log_fv.txt", "a") as textfile: 
        textfile.write(str(input) + "\n") 
    textfile.close()

'''
DATASET
'''
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
    
'''
DATASET2
'''
class ONEPW_Dataset2(Dataset):
    def __init__(self, neg16_img, neg8_img, zero_img, pos8_img, pos16_img, onepw_img,):
        '''
        data - train data path
        enh_img - train enhanced images path
        '''
        # self.train_data = data
        self.train_neg16_img = neg16_img
        self.train_neg8_img = neg8_img
        self.train_zero_img = zero_img
        self.train_pos8_img = pos8_img
        self.train_pos16_img = pos16_img

        self.train_onepw_img = onepw_img

        # self.images = sorted(os.listdir(self.train_data))

        self.neg16_images = sorted(os.listdir(self.train_neg16_img))
        self.neg8_images = sorted(os.listdir(self.train_neg8_img))
        self.zero_images = sorted(os.listdir(self.train_zero_img))
        self.pos8_images = sorted(os.listdir(self.train_pos8_img))
        self.pos16_images = sorted(os.listdir(self.train_pos16_img))

        self.onepw_images = sorted(os.listdir(self.train_onepw_img))
  
    #regresar la longitud de la lista, cuantos elementos hay en el dataset
    def __len__(self):
        if self.onepw_images is not None:
          assert len(self.neg16_images) == len(self.onepw_images), 'not the same number of images ans enh_images'
        return len(self.neg16_images)

    def __getitem__(self, idx):
        neg16_image_name = os.path.join(self.train_neg16_img, self.neg16_images[idx])
        neg16_image = np.load(neg16_image_name)
        neg16_image = torch.Tensor(neg16_image)
        neg16_image = neg16_image.permute(2, 0, 1)

        neg8_image_name = os.path.join(self.train_neg8_img, self.neg8_images[idx])
        neg8_image = np.load(neg8_image_name)
        neg8_image = torch.Tensor(neg8_image)
        neg8_image = neg8_image.permute(2, 0, 1)
        print("SHAPE neg8", neg8_image.shape)

        zero_image_name = os.path.join(self.train_zero_img, self.zero_images[idx])
        zero_image = np.load(zero_image_name)
        zero_image = torch.Tensor(zero_image)
        zero_image = zero_image.permute(2, 0, 1)

        pos8_image_name = os.path.join(self.train_pos8_img, self.pos8_images[idx])
        pos8_image = np.load(pos8_image_name)
        pos8_image = torch.Tensor(neg16_image)
        pos8_image = pos8_image.permute(2, 0, 1)

        pos16_image_name = os.path.join(self.train_pos16_img, self.pos16_images[idx])
        pos16_image = np.load(pos16_image_name)
        pos16_image = torch.Tensor(pos16_image)
        pos16_image = pos16_image.permute(2, 0, 1)

        print("SHAPE pos16", pos16_image.shape)
        
        # Concatenate the tensors along the first dimension
        combined_input_tensor = torch.cat((neg16_image, neg8_image, zero_image, pos8_image, pos16_image), dim=0)

        onepw_image_name = os.path.join(self.train_onepw_img, self.onepw_images[idx])
        onepw_img = np.load(onepw_image_name)
        onepw_img = torch.Tensor(onepw_img)
        onepw_img = onepw_img.unsqueeze(0)
        new_min = -1
        new_max = 1
        onepw_img = onepw_img * (new_max - new_min) + new_min

        return combined_input_tensor, onepw_img

def main():
    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    print(device)
    # save_dir = Path(os.getcwd())/'weights'/'v13'
    save_dir = '/mnt/nfs/efernandez/trained_models/DDPM_model/FV'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # training hyperparameters
    batch_size = 4  # 4 for testing, 16 for training
    n_epoch = 500
    l_rate = 1e-5  # changing from 1e-5 to 1e-6, new lr 1e-7

    # Loading Data
    # input_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\input_overfit'
    # output_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\target_overfit'
    # input_folder = r'C:\Users\u_imagenes\Documents\smerino\training\input'
    # output_folder = r'C:\Users\u_imagenes\Documents\smerino\training\target_enh'


    # train_dataset = ONEPW_Dataset(TRAIN_PATH, TRAIN_75PW_PATH)

    train_dataset = ONEPW_Dataset2(TRAIN_75PW_16m_PATH, TRAIN_75PW_8m_PATH, TRAIN_75PW_0_PATH, TRAIN_75PW_8p_PATH, TRAIN_75PW_16p_PATH, TRAIN_75PW_PATH)

    BATCH_SIZE = 4

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)

    #######################
    # dataset = gd.CustomDataset(input_folder, output_folder, transform=True)
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Dataloader length: {len(train_loader)}')
    #######################

    for i, (x, y) in enumerate(train_loader):
        print(i, x.shape,y.shape)
        if i==9: break

    # DDPM noise schedule
    time_steps = 100
    betas = gd.get_named_beta_schedule('linear', time_steps)
    diffusion = gd.SpacedDiffusion(
        use_timesteps = gd.space_timesteps(time_steps, section_counts=[time_steps]),
        betas = betas,
        model_mean_type = gd.ModelMeanType.EPSILON,
        model_var_type= gd.ModelVarType.FIXED_LARGE,
        loss_type = gd.LossType.MSE,
        rescale_timesteps = True,
    )
    
    # Model and optimizer
    nn_model = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
    # nn_model = UNETv10_5_2(emb_dim=64*4).to(device)
    print("Num params: ", sum(p.numel() for p in nn_model.parameters() if p.requires_grad))

    optim = torch.optim.Adam(nn_model.parameters(), lr=l_rate)

    trained_epochs = 0
    if trained_epochs > 0:
        nn_model.load_state_dict(torch.load(save_dir+f"/model_{trained_epochs}.pth", map_location=device))  # From last model
        loss_arr = np.load(save_dir+f"/loss_{trained_epochs}.npy").tolist()  # From last model
    else:
        loss_arr = []

    # Training
    nn_model.train()
    # pbar = tqdm(range(trained_epochs+1,n_epoch+1), mininterval=2)
    print(f' Epoch {trained_epochs}/{n_epoch}, {datetime.now()}')
    write_to_file('Training')
    write_to_file(datetime.now())
    for ep in range(trained_epochs+1, n_epoch+1):
        # pbar = tqdm(train_loader, mininterval=2)
        for x, y in train_loader:  # x: images
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # perturb data
            noise = torch.randn_like(y)
            t = torch.randint(0, time_steps, (x.shape[0],)).to(device)
            y_pert = diffusion.q_sample(y, t, noise)
            
            # use network to recover noise
            predicted_noise = nn_model(x, y_pert, t)

            # loss is mean squared error between the predicted and true noise
            loss = func.mse_loss(predicted_noise, noise)
            loss.backward()

            # nn.utils.clip_grad_norm_(nn_model.parameters(),0.5)
            loss_arr.append(loss.item())
            optim.step()
            # torch.save(nn_model.state_dict(), save_dir+f"/model_{ep}.pth")
            # write_to_file('Batch time:')
            # write_to_file(str(datetime.now()))

        print(f' Epoch {ep:03}/{n_epoch}, loss: {loss_arr[-1]:.2f}, {datetime.now()}')
        write_to_file('Epoch '+str(ep))
        write_to_file(str(datetime.now()))
        # save model every x epochs
        if ep % 20 == 0 or ep == int(n_epoch) or ep == 1:
            torch.save(nn_model.state_dict(), save_dir+f"/model_{ep}.pth")
            np.save(save_dir+f"/loss_{ep}.npy", np.array(loss_arr))

if __name__ == '__main__':
    main()
