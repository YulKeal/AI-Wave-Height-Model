"""
This code implements a model using the based on EarthFormer
architecture for predicting Significant Wave Height (SWH)
in the Global Sea region based on wind fields.

The model includes the following functionalities:

1. Model Training:
   - The module provides functions to train the EarthFormer
   model using wind field data and corresponding SWH
   measurements from the Global region.
   -The training process was run on an Ubuntu system with
    200GB of RAM and RTX 4090 GPU.

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset
import numpy as np
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time as tm
from torch.utils.data import Subset
import xarray as xr
import argparse
import random
from torch.nn.parallel import DataParallel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from earthformer.config import cfg
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from omegaconf import OmegaConf
from data_parallel import BalancedDataParallel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.linear_model import LinearRegression

def count_model_params(model):
    """
    This function calculates the total number of parameters and
    the number of trainable parameters in a PyTorch model.
    """

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def data_preprocess(dataset_name = "2022"):
    """
    Preprocesses the input dataset for swh prediction.

    Args:
        dataset_name (str): The name of the dataset to preprocess.
         Defaults to "2020" test set.

    Returns:
        tuple: A tuple containing the preprocessed data
        and metadata.
            - wind_u (ndarray): Wind velocity component 'u'.
            - wind_v (ndarray): Wind velocity component 'v'.
            - wave_height (ndarray): Wave height data.
            - lat (ndarray): Latitude values.
            - lon (ndarray): Longitude values.
    Notes:
        - The wind and wave field data in the nc file is sourced from ERA5
         reanalysis dataset.
    """

    data = xr.open_dataset('E:/Era5-Global-0.5/' + dataset_name + '.nc')
  #  print(data)
    lon = data['longitude'].data[::]
    lat = data['latitude'].data[::]
    wind_u = data['u10'].data
    wind_v = data['v10'].data
    wave_height = data['swh'].data
    return wind_u[:,:,:],wind_v[:,:,:],wave_height[:,:,:],lat,lon


class DynamicDataset(Dataset):
    """
        Dataset class for dynamically creating wind field data 240 hours
         before the forecast moment and SWH labels data for training.

        Args:
            wind_u (ndarray): Array containing wind velocity component 'u'.
            wind_v (ndarray): Array containing wind velocity component 'v'.
            wave_height (ndarray): Array containing wave height data.
            time (int): Length of each wind sequence. Defaults to 240 hours.

        Methods:
            __len__(): Returns the size of the dataset.
            __getitem__(idx): Returns historical wind field data and corresponding
             wave height label at the specified index.
        """
    def __init__(self, wind_u, wind_v, wave_height, time=240):
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.wave_height = wave_height



        self.sequence_length = time

        self.dataset_size = len(wind_u) - time
        print(self.dataset_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        indices = list(range(idx + self.sequence_length - 1, idx - 1, -1))

        wind_u_data = self.wind_u[indices]
        wind_v_data = self.wind_v[indices]
        wave_height_data = self.wave_height[idx + self.sequence_length]


        extend_edge = 20

        data = np.empty((2, self.sequence_length, lat_wind, lon_wind+ extend_edge*2), dtype=np.float32)

        data[0, :, :, extend_edge:-extend_edge] = wind_u_data
        data[1, :, :, extend_edge:-extend_edge] = wind_v_data
        data[0, :, :, 0:extend_edge] = wind_u_data[:, :, -extend_edge:]
        data[1, :, :, 0:extend_edge] = wind_v_data[:, :, -extend_edge:]
        data[0, :, :, -extend_edge:] = wind_u_data[:, :, :extend_edge]
        data[1, :, :, -extend_edge:] = wind_v_data[:, :, :extend_edge]

        data = np.transpose(data, (1, 2, 3,0))  # ????????
        label = np.nan_to_num(wave_height_data, nan=0, copy=False)

        data_tensor = torch.Tensor(data).float()
        label_tensor = torch.Tensor(label).float().unsqueeze(0).unsqueeze(-1)  # ????????????

        return data_tensor, label_tensor

class AdjustedDownsampleConvNet(nn.Module):
    """
    Data is downsampled through convolution and pooling layers before
    entering EarthFormer Model
    """
    def __init__(self):
        super(AdjustedDownsampleConvNet, self).__init__()
        self.conv1 = nn.Conv3d(2, 32, kernel_size=3, stride=2, padding=1)  # ????????????
        self.pool = nn.MaxPool3d(2, stride=2)  # ???????????
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)  # ???????????
        self.conv3 = nn.Conv3d(64, 16, kernel_size=3, stride=1, padding=1)  # ??????????

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class AdjustedUpsampleConvNet(nn.Module):
    """
     Up-sampling the EarthFormer model output to recover to label size
    """
    def __init__(self, input_channels=16, output_channels=1):
        super(AdjustedUpsampleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(16, output_channels, kernel_size=3, padding=1)
        self.final_resize = nn.AdaptiveAvgPool2d((281, 720))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.upsample1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.upsample2(x))
        x = F.relu(self.conv3(x))
        x = self.final_conv(x)
        x = self.final_resize(x)
        return x





class CuboidGlobalSWHModule(nn.Module):
    """
    This class defines the structure of the EarthFormer model,
    the setting of configuration parameters and reading from the
    external yaml and the forward propagation of the entire model.
    """
    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super(CuboidGlobalSWHModule, self).__init__()
        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.adjusted_up_net = AdjustedUpsampleConvNet(input_channels=16, output_channels=1)
        self.adjusted_down_net = AdjustedDownsampleConvNet()

        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="conv"
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
            final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)


        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout

    def get_base_config(self, oc_from_file=None):
        """
        Get the base configuration for the EearthFormer model.
        This method constructs the base configuration for the EF model by merging
        various sub-configurations such as layout,optimization, logging, trainer,
        and model configurations.

        In addition, for the convenience of configuration, we directly use an external
        yaml as a configuration file,so the configuration after this function did not work.
        The external yaml must be provided.
        """
        oc = OmegaConf.create()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.model = self.get_model_config()
        if oc_from_file is None:
            raise ValueError("YAML configuration file is empty or not provided.")
        else:
            oc = OmegaConf.merge(oc, oc_from_file)
            print('Successfully read YAML configuration file!')
        return oc


    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        height = 64
        width = 64
        in_len = 10
        out_len = 10
        data_channels = 2
        cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'  # The method for initializing the first input of the decoder
        cfg.initial_downsample_type = "conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_scale = 2
        cfg.initial_downsample_conv_layers = 2
        cfg.final_upsample_conv_layers = 1
        cfg.checkpoint_level = 2
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @staticmethod
    def get_layout_config():
        oc = OmegaConf.create()
        oc.in_len = 60
        oc.out_len = 1
        oc.layout = "NTHWC"  # The layout of the data, not the model
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 8
        oc.micro_batch_size = oc.total_batch_size

        oc.method = "adamw"
        oc.lr = 1E-3
        oc.wd = 1E-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 50
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 5
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "Era5Global"
        oc.monitor_lr = True
        oc.monitor_device = False
        oc.track_grad_norm = -1
        cfg.use_wandb = False
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 10
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = 32
        return oc


    def forward(self, input):

        input=input.permute(0,4, 1, 2, 3)
        input=self.adjusted_down_net(input)

        input = input.permute(0, 2, 3,4,1)
        pred_seq = self.torch_nn_module(input)
        pred_seq = torch.squeeze(pred_seq,dim=1).permute(0, 3, 1, 2)

        pred_seq =  self.adjusted_up_net(pred_seq)
        pred_seq = torch.unsqueeze(pred_seq,dim=4)
        return pred_seq


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_ear5_global', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default='cfg_global_ef.yaml', type=str)

    return parser


def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state_dict keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            # Remove prefix
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def custom_loss(outputs, labels, lat_cos):
    lat_cos = lat_cos.reshape(1,1, lat,1, 1)
    mse_loss = F.mse_loss(outputs, labels, reduction='none')
    corrected_loss = mse_loss * lat_cos
    corrected_loss_mean = corrected_loss.mean()
    return corrected_loss_mean

def NetTrain(epochs=20,batch=12,learning_rate = 0.001):

    parser = get_parser()
    args = parser.parse_args()
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0

    model= CuboidGlobalSWHModule(
        total_num_steps=epochs,
        save_dir=args.save,
        oc_file=args.cfg)

    writer = SummaryWriter(log_dir="./logs/" + datetime.now().strftime("%Y%m%d%H%M%S") + '_Earthnet' + ps)


    latest_checkpoint_path = os.path.join(model_path, "latest_checkpoint.pt")

    if os.path.exists(latest_checkpoint_path):
        print("Loading from latest checkpoint:", latest_checkpoint_path)
    else:

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        latest_epoch = -1
        for file_name in os.listdir(model_path):
            if not file_name.endswith(".pt"):
                continue
            try:
                epoch, loss = file_name.rsplit("_", 1)[0], file_name[:-3].split("_")[-1]  # ?????epoch?loss???
                epoch = int(epoch)
                loss = float(loss)
            except ValueError:
                continue
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint_path = os.path.join(model_path, file_name)

        if latest_epoch == -1:
            print("No valid checkpoint found in", model_path)
            latest_checkpoint_path = None
        else:
            print("Loading from checkpoint:", latest_checkpoint_path)

    if latest_checkpoint_path is not None:
        checkpoint = torch.load(latest_checkpoint_path)
        print(latest_checkpoint_path)
        if 'model' in checkpoint:
            model = checkpoint['model'].to(device)
        else:
            model = CuboidGlobalSWHModule(
                total_num_steps=epochs,
                save_dir=args.save,
                oc_file=args.cfg)
            model = model.to(device)
            print("Error: cannot find model in checkpoint!")

        model.load_state_dict( remove_module_prefix(checkpoint['model_state_dict']))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print("Loaded checkpoint {}: epoch={}, loss={}".format(latest_checkpoint_path, start_epoch, loss  ))

    else:
        start_epoch = 0
        net = model.to(device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        gpu0_bsz = 0
        acc_grad = 1
        model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).cuda()

    mse = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1) #not use

    total_params, trainable_params = count_model_params(model)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    model.train()
    years = list(range(2000, 2018))

    random.shuffle(years)


    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0

    loaded = np.load('mask_global_HRSWH.npz')
    mask= loaded['mask']
    mask= np.logical_not(mask)
    mask = mask[ np.newaxis, :, :, np.newaxis]
    mask=torch.tensor(mask).to(device)

    lat_start, lat_end = -70, 70
    lat_size = 281
    lat_cos_fix = np.cos(np.deg2rad(np.linspace(lat_start, lat_end, lat_size))).reshape(-1, 1)
    lat_cos_fix = torch.tensor(lat_cos_fix).to(device)

    for epoch in range(0,epochs):
        for year in years:

            running_loss = 0.0
            true_loss = 0.0
            model = model.to(device)

            wind_u, wind_v, wave_height, _, _ = data_preprocess(str(year))

            dataset = DynamicDataset(wind_u, wind_v, wave_height)

            dataloader = DataLoader(dataset, batch_size=batch, shuffle=True,drop_last=True, num_workers=10)
            dataloader_length = len(dataloader)

            print(f"Epoch:{epoch + 1},Year:{year},DataShape{dataloader_length}")

            for data, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False):

                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)

                outputs = torch.masked_fill(outputs, mask, 0)
                labels = torch.masked_fill(labels, mask, 0)

                loss=custom_loss(outputs,labels,lat_cos_fix)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    running_loss += loss.item()
                    tqdm.write(f"MSE: {loss.item():.4f}", end="")

                    true_loss += (torch.mean((outputs - labels) ** 2))

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{epochs}, Rmse Loss: {np.sqrt(running_loss / dataloader_length)},True Loss:{true_loss / dataloader_length}, Current Learning Rate: {current_lr}")
            writer.add_scalar('Learning Rate:', current_lr, global_step=epoch)
            writer.add_scalar('Train Rmse Loss:', np.sqrt(running_loss / dataloader_length), global_step=epoch)
            if epoch % 1 == 0:
                writer.add_image('predict', outputs[0, :].squeeze(3), global_step=epoch)
                writer.add_image('label', labels[0, :].squeeze(3), global_step=epoch)

                rmse = torch.sqrt(torch.mean(torch.square(outputs - labels), 0)).detach().cpu().numpy().squeeze()

                plt.figure()
                cax = plt.matshow(rmse, cmap='viridis')
                plt.colorbar(cax)

                writer.add_figure('RMSE Heatmap', plt.gcf(), global_step=epoch)
            del dataset, dataloader,data,labels

            save_dict = {}
            save_dict['epoch'] = epoch
            save_dict['model_state_dict'] = model.state_dict()
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_dict['loss'] = np.sqrt(running_loss / dataloader_length)


            checkpoint_path = os.path.join(model_path, "latest_checkpoint.pt")

            torch.save(save_dict, checkpoint_path)

            print("Model saved to {}".format(checkpoint_path))

        if epoch % 1 == 0:

            # validation
            print("validation")
            wind_u, wind_v, wave_height, _, _ = data_preprocess('2022')

            validation_dataset = DynamicDataset(wind_u, wind_v, wave_height)
            validation_dataloader = DataLoader(validation_dataset, batch_size=batch, shuffle=True,drop_last=True,num_workers=10)

            validation_loss = 0
            real_loss = 0
            validation_step = 0
            rmse = np.zeros([lat, lon])
            model.eval()
            with torch.no_grad():
                for validation_data, target in tqdm(validation_dataloader):
                    validation_data, target = validation_data.to(device), target.to(device)

                    logits = model(validation_data)
                    logits = torch.masked_fill(logits, mask, 0)
                    target = torch.masked_fill(target, mask, 0)

                    msevalue = mse(logits, target).item()

                    realvalue = torch.mean(torch.abs(logits.squeeze() - target.squeeze()))
                    validation_loss += msevalue
                    real_loss += realvalue
                    rmse += torch.sqrt(
                        torch.mean(torch.square(logits.squeeze(4).squeeze(1) - target.squeeze(4).squeeze(1)),
                                   0)).detach().cpu().numpy()

                    validation_step += 1

                validation_loss /= validation_step
                real_loss /= validation_step
                rmse /= validation_step
                del validation_dataset, validation_dataloader,validation_data,target,logits
                print(
                    '\n  Epoch: {} validation set: Average Mse loss: {:.6f},Average RMse loss: {:.6f}, Abs loss: {:.6f}'.format(
                        epoch, validation_loss, np.sqrt(validation_loss), real_loss))
                writer.add_scalar('validation Rmse Loss:', np.sqrt(validation_loss), global_step=epoch)
                plt.figure()
                cax = plt.matshow(rmse, cmap='viridis')
                plt.colorbar(cax)
                writer.add_figure('RMSE validation Heatmap', plt.gcf(), global_step=epoch)
                plt.close()

                if np.sqrt(validation_loss) < best_loss:
                    best_loss = np.sqrt(validation_loss)
                    epochs_no_improve = 0

                else:
                    epochs_no_improve += 1

            save_dict = {}
            save_dict[f'epoch'] = epoch
            save_dict[f'model_state_dict'] = model.state_dict()
            save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
            save_dict[f'loss'] = np.sqrt(validation_loss)

            checkpoint_path = os.path.join(model_path,
                                           "{}_{}_mid_".format(epoch, (running_loss / dataloader_length)) + ps + ".pt")
            torch.save(save_dict, checkpoint_path)
            print("Save intermediate checkpoints")

    save_dict = {}
    save_dict[f'epoch'] = epoch
    save_dict[f'model_state_dict'] = model.state_dict()
    save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
    save_dict[f'loss'] = np.sqrt(validation_loss)

    checkpoint_path = os.path.join(model_path, "{}_{}.pt".format(epoch, (running_loss / dataloader_length)))
    torch.save(save_dict, checkpoint_path)
    print("Training completed")



in_channels = 2
lat_wind =281
lon_wind =720
lat = 281
lon =720


model_path=r"./model"
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ps="era5_global"

NetTrain(epochs=20,batch=12,learning_rate = 0.001)
