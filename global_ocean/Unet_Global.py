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
import copy
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
#from sklearn.linear_model import LinearRegression

def data_preprocess(dataset_name = "southocean_2022"):
    data = xr.open_dataset('E:/Era5-Global-0.5/' + dataset_name + '.nc')

    lon = data['longitude'].data[::]
    lat = data['latitude'].data[::]


    wind_u = data['u10'].data
    wind_v = data['v10'].data


    wave_height = data['swh'].data

    return wind_u[:,:,:],wind_v[:,:,:],wave_height[:,:,:],lat,lon





class DynamicDataset(Dataset):
    def __init__(self, wind_u, wind_v, wave_height, time=240):
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.wave_height = wave_height
        self.time = time

        #
        # self.intervals = [(0, 24, 2), (27, 72, 4), (76, 144, 6), (150, 240, 4)]
        # self.sequence_length = sum((b - a) // c + 1 for a, b, c in self.intervals)

        self.sequence_length = 240

        self.dataset_size = len(wind_u) - time
        print(self.dataset_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):


        # indices = []
        # for start, end, step in self.intervals:
        #      indices.extend(range(start, end + 1, step))
        # indices = [(idx + self.time - 1) - element for element in indices]

        indices = list(range(idx + self.time - 1, idx - 1, -1))


        wind_u_data = self.wind_u[indices]
        wind_v_data = self.wind_v[indices]
        wave_height_data = self.wave_height[idx + self.time]


        extend_edge = 20

        data = np.empty((2, self.sequence_length, lat_wind, lon_wind+ extend_edge*2), dtype=np.float32)

        data[0, :, :, extend_edge:-extend_edge] = wind_u_data
        data[1, :, :, extend_edge:-extend_edge] = wind_v_data
        data[0, :, :, 0:extend_edge] = wind_u_data[:, :, -extend_edge:]
        data[1, :, :, 0:extend_edge] = wind_v_data[:, :, -extend_edge:]
        data[0, :, :, -extend_edge:] = wind_u_data[:, :, :extend_edge]
        data[1, :, :, -extend_edge:] = wind_v_data[:, :, :extend_edge]


        label = np.nan_to_num(wave_height_data, nan=0, copy=False)


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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final_resize = nn.AdaptiveAvgPool2d((281, 720))

    def forward(self, x):
        x=self.conv(x)
        x=self.final_resize(x)
        return x



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)



    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # Concatenate along the channels axis
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

def custom_loss(outputs, labels, lat_cos):

    lat_cos = lat_cos.reshape(1, lat,1)

    mse_loss = F.mse_loss(outputs, labels, reduction='none')


    corrected_loss = mse_loss * lat_cos


    corrected_loss_mean = corrected_loss.mean()

    return corrected_loss_mean

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            # Remove prefix
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def NetTrain():


    in_channels = 2

    learning_rate=0.0001
    epochs=10
    batch=2
    sequence_length=240

    model = WNet(n_vars=2,n_times=sequence_length, n_classes=1)

    writer = SummaryWriter(log_dir="./logs/" + datetime.now().strftime("%Y%m%d%H%M%S") + '_Unet' + ps)
    fake_img = torch.zeros((1, sequence_length, lat, lon))
    writer.add_graph(model, [fake_img,fake_img])


    latest_checkpoint_path = None

    latest_epoch = -1
    start_epoch = 0

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
            model = WNet(n_vars=2,n_times=sequence_length, n_classes=1)
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
    from data_parallel import BalancedDataParallel



    mse = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)




    model.train()
    years = list(range(2000,2006))
    random.shuffle(years)

###############################Early Stopping######################
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0


    loaded = np.load('mask_global_HRSWH.npz')
    mask= loaded['mask']
    mask= np.logical_not(mask)
    mask = mask[:, :]
    mask=torch.tensor(mask).to(device)


    lat_start, lat_end = -70, 70
    lat_size = 281
    lat_cos_fix = np.cos(np.deg2rad(np.linspace(lat_start, lat_end, lat_size))).reshape(-1, 1)
    lat_cos_fix = torch.tensor(lat_cos_fix).to(device)


    for epoch in range(1,epochs):
        for year in years:
            running_loss = 0.0
            true_loss=0.0
            model = model.to(device)

            wind_u, wind_v, wave_height, _, _ = data_preprocess(str(year))
            dataset = DynamicDataset(wind_u, wind_v, wave_height)

            dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True,num_workers=2)
            dataloader_length = len(dataloader)

       
            print(f"Epoch:{epoch+1},Year:{year},DataShape{dataloader_length }")




            del wind_u, wind_v, wave_height

            for data, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data[:, 0, :], data[:, 1, :])  # torch.Size([20, 72, 14, 32])


                outputs = torch.masked_fill(outputs, mask, 0)
                labels = torch.masked_fill(labels, mask, 0)
                #loss = mse(outputs, labels)
                loss = custom_loss(outputs, labels, lat_cos_fix)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tqdm.write(f"MSE: {loss.item():.4f}", end="")
                # mask = (labels != 0).float()
                true_loss += (torch.mean((outputs - labels) ** 2))

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{epochs}, Rmse Loss: {np.sqrt(running_loss /  dataloader_length)},True Loss:{true_loss /  dataloader_length}, Current Learning Rate: {current_lr}")
            writer.add_scalar('Learning Rate:', current_lr, global_step=epoch)
            writer.add_scalar('Train Rmse Loss:', np.sqrt(running_loss /  dataloader_length), global_step=epoch)
            if epoch % 10 == 0:
                writer.add_image('predict', outputs[0, :].unsqueeze(0), global_step=epoch)
                writer.add_image('label', labels[0, :].unsqueeze(0), global_step=epoch)

                rmse = torch.sqrt(torch.mean(torch.square(outputs - labels), 0)).detach().cpu().numpy()

                # ??Matplotlib?????RMSE
                plt.figure()
                cax = plt.matshow(rmse, cmap='viridis')  # ??'viridis' colormap????????????colormap
                plt.colorbar(cax)

                # ?Matplotlib?????TensorBoard
                writer.add_figure('RMSE Heatmap', plt.gcf(), global_step=epoch)
            del dataset,dataloader

            save_dict = {}
            save_dict['epoch'] = epoch
            save_dict['model_state_dict'] = model.state_dict()
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_dict['loss'] = np.sqrt(running_loss / dataloader_length)

            checkpoint_path = os.path.join(model_path, "latest_checkpoint.pt")

            torch.save(save_dict, checkpoint_path)

            print("Model saved to {}".format(checkpoint_path))

        if epoch % 1 == 0:



            # test
            print("test")
            wind_u, wind_v, wave_height, _, _ = data_preprocess('2022')

            test_dataset = DynamicDataset(wind_u, wind_v, wave_height)
            test_dataloader = DataLoader(test_dataset, batch_size=batch,num_workers=2)

            del  wind_u, wind_v, wave_height

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
                    target= torch.masked_fill( target, mask, 0)
                    # ?? mask??? labels ??? 0 ???
                   # mask = (target != 0)
                    # ?? mask ? outputs?? outputs ??? mask ? True ?????? 0
                    #logits = logits  * mask
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
               # rmse = torch.sqrt(torch.mean(torch.square(logits - target), 0)).detach().cpu().numpy()


                plt.figure()
                cax = plt.matshow(rmse, cmap='viridis')  # ??'viridis' colormap????????????colormap
                plt.colorbar(cax)
                # ?Matplotlib?????TensorBoard
                writer.add_figure('RMSE Test Heatmap', plt.gcf(), global_step=epoch)
                plt.close()

                if np.sqrt(test_loss) < best_loss:
                    best_loss = np.sqrt(test_loss)
                    epochs_no_improve = 0

                else:
                    epochs_no_improve += 1

     #   if epochs_no_improve == patience and epoch>20:
      #      print(f'Early stopping after {epoch + 1} epochs.')
       #     break

        if epoch%1==0 :
            save_dict = {}
            save_dict[f'epoch'] =  epoch
            save_dict[f'model_state_dict'] =model.state_dict()
            save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
            save_dict[f'loss'] = np.sqrt(test_loss)

            checkpoint_path = os.path.join(model_path, "{}_{}_mid_".format(epoch, (running_loss /  dataloader_length))+ps+".pt")
            torch.save(save_dict, checkpoint_path)





    save_dict = {}
    save_dict[f'epoch'] =  epoch
    save_dict[f'model_state_dict'] =model.state_dict()
    save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
    save_dict[f'loss'] = np.sqrt(test_loss)

    checkpoint_path = os.path.join(model_path, "{}_{}.pt".format(epoch, (running_loss /  dataloader_length)))
    torch.save(save_dict, checkpoint_path)




def continuous_inference(model, test_dataloader):
    total_time = int(len(test_dataloader))

    rmse_figure_data = np.zeros([2, total_time, lat, lon])
    out_data = []
    label_data = []

    count = 0



    with torch.no_grad():
        for test_data, target in tqdm(test_dataloader):
            test_data, target = test_data.to(device), target.to(device)

            out = model(test_data[:,0,:],test_data[:,1,:])  # ??????
            out=torch.unsqueeze(out,0)


            out[target == 0] = 0
            out[out<0] = 0


            labels=target


    

            loss = torch.mean(torch.square(out - labels))

            varnumber = 0
            out_data.append(
                np.squeeze(np.reshape(out.detach().cpu().numpy(), [lat, lon])))
            label_data.append(np.squeeze(np.reshape(labels.detach().cpu().numpy(), [lat, lon])))

            rmse_figure_data[0, count, :, :] = np.reshape(out.detach().cpu().numpy(), [lat, lon])  # predict:0
            rmse_figure_data[1, count, :, :] = np.reshape(labels.detach().cpu().numpy(), [lat, lon])  # label:1

            print(' from {} hours inference {} step Mse Loss: {:.6f}  '.format(count, 1, loss))

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



    # checkpoint = torch.load(model_path + '/1_0.004565339488434782_mid_era5_unet_global_.pt')
    #
    # model.load_state_dict(checkpoint['model_state_dict'])
    #
    # torch.save(model, "D:\GitHub Code Upload\HuggingFace\global_unet.pt")
    model = torch.load("D:\GitHub Code Upload\HuggingFace\含模型\global_ocean\global_unet.pt")

    model.eval()



    for year in range(2020,2021): 

        print(year)

        wind_u, wind_v, wave_height, axis_lat, axis_lon = data_preprocess(str(year))
        #        data, labels = create_dataset(wind_u, wind_v, wave_height,time)
        dataset = DynamicDataset(wind_u, wind_v, wave_height)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        lat = axis_lat.shape[0]
        lon = axis_lon.shape[0]

 
        num_steps = 1  

        rmse_figure_data, rrmse_figure_data, correlation_coefficients, bias, out_data, label_data = continuous_inference(
            model, dataloader)
        print("save")

        np.savez('/media/ubuntu/Results/'+'plot_' + ps + '_' + str(year) + '.npz',
                 rmse_figure_data=rmse_figure_data,
                 rrmse_figure_data=rrmse_figure_data,
                 correlation_coefficients=correlation_coefficients,
                 bias=bias,
                 out_data=out_data,
                 label_data=label_data,
                 axis_lat=axis_lat,
                 axis_lon=axis_lon)

        print("plot")
        plotfig.Plot_Scatter(out_data, label_data, time_step=None, filename='/media/ubuntu/Results/'+str(year)+ '_' +ps+ '_sc')
        plotfig.Plot_RMSE_N(rmse_figure_data, axis_lat, axis_lon, filename='/media/ubuntu/Results/'+str(year) + '_'+ps+ '_rmse')
        plotfig.Plot_RMSE_N(correlation_coefficients, axis_lat, axis_lon, filename='/media/ubuntu/Results/'+str(year) + '_'+ps+ '_cor', figurename='Spatial Correlation Map')
        plotfig.Plot_RMSE_N(rrmse_figure_data, axis_lat, axis_lon, filename='/media/ubuntu/Results/'+str(year)+ '_' +ps+ '_rrmse', figurename='Relative RMSE Spatial Distribution Map')
        plotfig.Plot_RMSE_N(bias, axis_lat, axis_lon, filename='/media/ubuntu/Results/'+str(year)+ '_' +ps+ '_bias', figurename='Bias Spatial Distribution Map')


        del rmse_figure_data, rrmse_figure_data,correlation_coefficients, out_data, label_data, wind_u, wind_v, wave_height, axis_lat, axis_lon






os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


in_channels = 2  
lat_wind =281 
lon_wind =720 
lat = 281 
lon =720 

#time = 72  
model_path=r"./unet_global_model"
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cuda:0"
print(device)



model = WNet(n_vars=2,n_times=240, n_classes=1)
model=model.to(device)
print(model)
ps="era5_unet_global_"

print(model)
NetTrain()
#NetInference()
