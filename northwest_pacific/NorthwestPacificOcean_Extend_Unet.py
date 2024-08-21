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
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from scipy.ndimage import zoom
from datetime import datetime

def data_preprocess(dataset_name = "2022"):
    data = xr.open_dataset('E:/Era5-Global-0.5/' + dataset_name + '.nc')
    lon = data['longitude'].data[::]
    lat = data['latitude'].data[::]

    lon_min = np.where(lon == 100)[0][0]
    lon_max = np.where(lon == 135)[0][0] + 1
    lat_min = np.where(lat == 45)[0][0]
    lat_max = np.where(lat == 0)[0][0] + 1

    wave_height = data['swh'].data[:, lat_min:lat_max, lon_min:lon_max]

    lon_min_wind = np.where(lon == 90)[0][0]
    lon_max_wind = np.where(lon == 179.5)[0][0] + 1
    lat_min_wind = np.where(lat == 50)[0][0]
    lat_max_wind = np.where(lat == -10)[0][0] + 1

    lon = lon[lon_min:lon_max]
    lat = lat[lat_min:lat_max]

    wind_u = data['u10'].data[:, lat_min_wind:lat_max_wind, lon_min_wind:lon_max_wind]
    wind_v = data['v10'].data[:, lat_min_wind:lat_max_wind, lon_min_wind:lon_max_wind]

    return wind_u, wind_v, wave_height, lat, lon


class DynamicDataset(Dataset):
    def __init__(self, wind_u, wind_v, wave_height, time=120):
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.wave_height = wave_height
        self.time = time

        self.sequence_length = time

        self.dataset_size = len(wind_u) - time
        print(self.dataset_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):



        indices = list(range(idx + self.time - 1, idx - 1, -1))

        wind_u_data = self.wind_u[indices]
        wind_v_data = self.wind_v[indices]
        wave_height_data = self.wave_height[idx + self.time]

        data = np.empty((2, self.sequence_length, wind_lat, wind_lon), dtype=np.float32)



        data[0, :, :,:] = wind_u_data
        data[1, :, :, :] = wind_v_data
        label = np.nan_to_num(wave_height_data, nan=0)


        data = torch.Tensor(data).float()
        label = torch.Tensor(label).float()


        return data, label




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),

        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Timefold(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.timesfold = nn.Sequential(

             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
             nn.SiLU(inplace=True),
             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.timesfold(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3,padding=1)
        self.silu= nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(16, out_channels,kernel_size=3, padding=1)



    def forward(self, x):
        x=self.conv(x)
        x=self.silu(x)
        x= self.conv1(x)
        x = self.silu(x)
        x=x[:,:,10:-20,20:-89]
        return  x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)



    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class WNet(nn.Module):
    def __init__(self, n_vars, n_times, n_classes, bilinear=True):
        super(WNet, self).__init__()
        self.n_vars = n_vars
        self.n_times = n_times
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.timefold = Timefold(n_times, int(n_times / 2))
        self.inc = DoubleConv(n_times, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.silu=nn.SiLU

    def forward(self, x_u, x_v):

        x_u = self.timefold(x_u)
        x_v = self.timefold(x_v)
        x1 = self.inc(torch.cat([x_u, x_v], dim=1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = torch.squeeze(logits)

        return logits


def custom_loss(outputs, labels):
    mask = (labels != 0).float()
    loss = torch.sum(mask * (outputs - labels) ** 2)

    return loss
criterion = custom_loss

def NetTrain():



    sequence_length=120
    time = sequence_length
    learning_rate=0.00001
    epochs=100
    batch=50


    wind_u, wind_v, wave_height, lat, lon = data_preprocess('2020')
    lat = lat.shape[0]
    lon = lon.shape[0]

    wave_height=np.nan_to_num(wave_height[0, :, :],nan=0)
    mask = torch.tensor(wave_height == 0)
    mask = torch.unsqueeze(mask, dim=0).to(device)

    model = WNet(n_vars=2,n_times=sequence_length, n_classes=1)


    writer = SummaryWriter(log_dir="./logs/" + datetime.now().strftime("%Y%m%d%H%M%S") + '_Unet' + ps)
    # fake_img = torch.zeros((1, time, wind_lat, wind_lon))
    latest_checkpoint_path = None
    latest_epoch = -1
    start_epoch = 0

    for file_name in os.listdir(model_path):
        if not file_name.endswith(".pt"):
            continue
        try:
            epoch, loss = file_name[:-3].split("_")
            epoch = int(epoch)
            loss = float(loss)
        except:
            continue
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_checkpoint_path = os.path.join(model_path, file_name)

    if latest_checkpoint_path is not None:
        checkpoint = torch.load(latest_checkpoint_path)
        print(latest_checkpoint_path)
        if 'model' in checkpoint:
            model = checkpoint['model'].to(device)
        else:
            model = WNet(n_vars=2,n_times=sequence_length, n_classes=1)
            model = model.to(device)
            print("Error: cannot find model in checkpoint!")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print("Loaded checkpoint {}: epoch={}, loss={}".format(latest_checkpoint_path, start_epoch, loss  ))
    else:
        start_epoch = 0
        net = model.to(device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    mse = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.1)
    import random
    model.train()
    years = list(range(2000, 2018))
    random.shuffle(years)
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0



    for epoch in range(epochs):
        for year in years:
            running_loss = 0.0
            true_loss=0.0
            model = model.to(device)

            wind_u, wind_v, wave_height, _, _ = data_preprocess(str(year))
            print(f"Epoch:{epoch+1},Year:{year}")

            dataset = DynamicDataset(wind_u, wind_v, wave_height)
            dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True,num_workers=4)


            dataloader_length=len(dataloader)

            for data, labels in tqdm(dataloader):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data[:, 0, :], data[:, 1, :])
                outputs = torch.masked_fill(outputs, mask, 0)
                loss = mse(outputs, labels)
                loss.backward()
                optimizer.step()


                with torch.no_grad():
                    running_loss += loss.item()
                    true_loss += (torch.sum((outputs - labels) ** 2)) / torch.sum(mask)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{epochs}, Rmse Loss: {np.sqrt(running_loss /  dataloader_length)},True Loss:{true_loss /  dataloader_length}, Current Learning Rate: {current_lr}")

            writer.add_scalar('Learning Rate:', current_lr, global_step=epoch)
            writer.add_scalar('Train RMSE Loss:', np.sqrt(running_loss / dataloader_length), global_step=epoch)


        if epoch % 1 == 0:


            del dataset,dataloader
            wind_u, wind_v, wave_height, _, _ = data_preprocess('2022')

            test_dataset = DynamicDataset(wind_u, wind_v, wave_height)
            test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=True, drop_last=True,num_workers=4)

            test_loss = 0
            real_loss = 0
            test_step = 0
            rmse = np.zeros([lat, lon])



            model.eval()
            with torch.no_grad():
          
                for test_data, target in test_dataloader:
                    test_data, target=test_data.to(device), target.to(device)

                    logits = model(test_data[:,0,:],test_data[:,1,:])
                    logits=torch.masked_fill(logits, mask, 0)


                    msevalue = mse(logits, target).item()


                    realvalue = torch.mean(torch.abs(logits - target))
                    test_loss += msevalue
                    real_loss += realvalue
                    rmse += torch.sqrt(torch.mean(torch.square(logits - target), 0)).detach().cpu().numpy()

                    test_step += 1


                test_loss /= test_step
                real_loss /= test_step
                rmse  /= test_step

                print(
                    '\n  Epoch: {} Test set: Average Mse loss: {:.6f},Average RMse loss: {:.6f}, Abs loss: {:.6f}'.format(
                         epoch, test_loss, np.sqrt(test_loss), real_loss))
                writer.add_scalar('Test Rmse Loss:', np.sqrt(test_loss), global_step=epoch)


                if np.sqrt(test_loss) < best_loss:
                    best_loss = np.sqrt(test_loss)
                    epochs_no_improve = 0

                else:
                    epochs_no_improve += 1

        if epoch%1==0 :
            save_dict = {}
            save_dict[f'epoch'] =  epoch
            save_dict[f'model_state_dict'] =model.state_dict()
            save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
            save_dict[f'loss'] = np.sqrt(test_loss)

            checkpoint_path = os.path.join(model_path, "{}_{}_mid_".format(epoch, (running_loss /  dataloader_length))+ps+".pt")
            torch.save(save_dict, checkpoint_path)
            print("Save")



    save_dict = {}
    save_dict[f'epoch'] =  epoch
    save_dict[f'model_state_dict'] =model.state_dict()
    save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
    save_dict[f'loss'] = np.sqrt(test_loss)

    checkpoint_path = os.path.join(model_path, "{}_{}.pt".format(epoch, (running_loss /  dataloader_length)))
    torch.save(save_dict, checkpoint_path)
    print("Save")



def continuous_inference(model, test_dataloader):
    total_time = int(len(test_dataloader))

    rmse_figure_data = np.zeros([2, total_time, lat, lon])
    out_data = []
    label_data = []

    count = 0
    with torch.no_grad():
        for test_data, target in tqdm(test_dataloader):
            test_data, target = test_data.to(device), target.to(device)

            out = model(test_data[:,0,:],test_data[:,1,:])
            out=torch.unsqueeze(out,0)


            out[target == 0] = 0


            labels=target

            loss = torch.mean(torch.square(out - labels))

            varnumber = 0
            out_data.append(
                np.squeeze(np.reshape(out.detach().cpu().numpy(), [lat, lon])))
            label_data.append(np.squeeze(np.reshape(labels.detach().cpu().numpy(), [lat, lon])))

            rmse_figure_data[0, count, :, :] = np.reshape(out.detach().cpu().numpy(), [lat, lon])
            rmse_figure_data[1, count, :, :] = np.reshape(labels.detach().cpu().numpy(), [lat, lon])

            count += 1
            if count == total_time: break;

    rmse_figure = np.sqrt(np.mean(np.square(rmse_figure_data[0, :] - rmse_figure_data[1, :]), 0))

    relative_rmse_figure = np.zeros([lat, lon])
    label_mean = np.mean(label_data, 0)
    relative_rmse_figure = rmse_figure / np.where(label_mean == 0, 1, label_mean)
    correlation_coefficients = np.zeros([lat, lon])
    bias=  np.zeros([lat, lon])

    bias=np.mean(rmse_figure_data[0, :] -rmse_figure_data[1, :],axis=0)
    for i in range(rmse_figure_data.shape[2]):
        for j in range(rmse_figure_data.shape[3]):
            predictions = rmse_figure_data[0, :, i, j]
            actuals = rmse_figure_data[1, :, i, j]
            if np.all(actuals == 0) == True:
                correlation_coefficients[i, j] = 1
                continue
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            correlation_coefficients[i, j] = correlation

    return rmse_figure,relative_rmse_figure, correlation_coefficients,bias, out_data, label_data




def NetInference():
    checkpoint = torch.load('northwest_pacific_extend_unet.pt')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for year in range(2000,2021):
        print(year)
        wind_u, wind_v, wave_height, axis_lat, axis_lon = data_preprocess(str(year))

        dataset = DynamicDataset(wind_u, wind_v, wave_height)

        del wind_u, wind_v, wave_height
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        lat = axis_lat.shape[0]
        lon = axis_lon.shape[0]
        num_steps = 1

        rmse_figure_data, rrmse_figure_data, correlation_coefficients, bias, out_data, label_data = continuous_inference(
            model, dataloader)

        results_path = './results/Unet/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        plot_path = './results/Unet/plot/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        np.savez(results_path + 'plot_' + ps + '_' + str(year) + '.npz',
                 rmse_figure_data=rmse_figure_data,
                 rrmse_figure_data=rrmse_figure_data,
                 correlation_coefficients=correlation_coefficients,
                 bias=bias,
                 out_data=out_data,
                 label_data=label_data,
                 axis_lat=axis_lat,
                 axis_lon=axis_lon)





os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
in_channels = 2

wind_lat = 121
wind_lon = 180

lat = 91
lon = 71

time = 120
model_path=r"./Unet_extend"+"/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = WNet(n_vars=2,n_times=time, n_classes=1)
model=model.to(device)
print(model)
ps="northwest_pacific_extend_unet"

#NetTrain()
NetInference()
