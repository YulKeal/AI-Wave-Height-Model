"""
This code implements a model using the U-Net architecture
for predicting Significant Wave Height (SWH) in
the Black Sea region based on wind fields.

The model includes the following functionalities:

1. Model Training:
   - The module provides functions to train the U-Net model
   using wind field data and corresponding SWH measurements
   from the Black Sea region.

2. Bias Correction with Linear Regression:
   - After training, the module allows for bias correction
   of both the training dataset and the model predictions
   at each location using linear regression.
   - It includes functions to perform linear regression on
   the training dataset and apply the obtained parameters
   to correct biases in the model predictions.

3. Inference with Regression Model:
   - The module enables inference using the trained U-Net
   model and the parameters (slope 'k' and intercept 'b')
   obtained from the linear regression model.
   - It provides functions to perform inference on new wind
   field data to predict SWH values, incorporating the bias
   correction based on the regression parameters.
"""

import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def data_preprocess(dataset_name="2020"):
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
        - The wind field data in the NPZ file is sourced from ERA5
         reanalysis dataset.
        - The wave height data in the NPZ file is sourced from Copernicus
         Monitoring Environment Marine's
        Black Sea Waves Analysis and Forecast dataset.
    """
    data = np.load('E:/WavePredict/NewDataSet/' + dataset_name + '.npz')
    axis = np.load('E:/WavePredict/NewDataSet/axis.npz')
    axis_lat = axis['lat'][::2]
    axis_lon = axis['lon'][::2]
    wind_u = data['U10']  # 输入本时刻风u
    wind_v = data['V10']  # 输入本时刻风v
    wave_height = data['hs']
    wave_height = np.flip(wave_height[:, ::2, ::2], axis=1)
    return wind_u, wind_v, wave_height, axis_lat, axis_lon


class DynamicDataset(Dataset):
    """
        Dataset class for dynamically creating wind field data 72 hours
         before the forecast moment and SWH labels data for training.

        Args:
            wind_u (ndarray): Array containing wind velocity component 'u'.
            wind_v (ndarray): Array containing wind velocity component 'v'.
            wave_height (ndarray): Array containing wave height data.
            time (int): Length of each wind sequence. Defaults to 72 hours.

        Methods:
            __len__(): Returns the size of the dataset.
            __getitem__(idx): Returns  wind field data and corresponding
             wave height label at the specified index.
        """

    def __init__(self, wind_u, wind_v, wave_height,time=72):
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

        extend_edge = 0

        data = np.empty((2, self.sequence_length, lat_wind,
                         lon_wind + extend_edge * 2), dtype=np.float32)

        data[0, :, :, :] = wind_u_data
        data[1, :, :, :] = wind_v_data
        label = np.nan_to_num(wave_height_data, nan=0)
        data = torch.Tensor(data).float()
        label = torch.Tensor(label).float()
        return data, label


class DoubleConv(nn.Module):
    """
    This class defines a double convolutional layer consisting
    of two 2D convolutional layers followed byactivation functions (SiLU).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #   nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #    nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    This class defines a downscaling module that consists
     of a max pooling layer followed by a double convolutional layer.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Timefold(nn.Module):
    """
    This class is mainly used to collapse the time dimension of
    the wind component of the input network
    """
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
    """
    This class is mainly used to collapse the time dimension of
    the wind component of the input network
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """
    This class defines an upsampling module that combines feature maps
    from the skip connection with the upsampled feature maps.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # Concatenate along the channels axis
        return self.conv(x)


class UNet(nn.Module):
    """
    This class defines the U-Net model architecture for SWH prediction
    """
    def __init__(self, n_vars, n_times, n_classes, bilinear=True):
        super(UNet, self).__init__()
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
        self.silu = nn.SiLU

    def forward(self, x_u, x_v):

        x_u = self.timefold(x_u)
        x_v = self.timefold(x_v)
        # Encoder part
        x1 = self.inc(torch.cat([x_u, x_v], dim=1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder part
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output
        return torch.squeeze(self.outc(x))


def custom_loss(outputs, labels):
    # 创建一个掩码，其中 labels 不等于0的位置为1，等于0的位置为0
    mask = (labels != 0).float()

    # 计算新的损失，只考虑非零值的部分
    loss = torch.sum(mask * (outputs - labels) ** 2)

    return loss


# 使用自定义损失
criterion = custom_loss  # 使用新的损失函数


def NetTrain():
    # 创建模型实例

    learning_rate = 0.0001
    epochs = 41
    batch = 30

    # validation
    wind_u, wind_v, wave_height, _, _ = data_preprocess('2022')

    validation_dataset = DynamicDataset(wind_u, wind_v, wave_height, time)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=batch, shuffle=True, drop_last=True)

    mask = torch.tensor(np.isnan(wave_height[0, :, :]))
    mask = torch.unsqueeze(mask, dim=0).to(device)

    model = UNet(n_vars=2, n_times=time, n_classes=1)

    writer = SummaryWriter(log_dir="./logs/" +
                                   datetime.now().strftime("%Y%m%d%H%M%S") + '_Unet' + info)
    fake_img = torch.zeros((1, time, lat, lon))
    writer.add_graph(model, [fake_img, fake_img])

    # 恢复最新的 checkpoint
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
            model = UNet(n_vars=2, n_times=time, n_classes=1)
            model = model.to(device)
            print("Error: cannot find model in checkpoint!")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print("Loaded checkpoint {}: epoch={}, loss={}".
              format(latest_checkpoint_path, start_epoch, loss))
    else:
        start_epoch = 0
        net = model.to(device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    #
    # model = model.to(device)

    # 定义损失函数和优化器
    #  criterion = nn.MSELoss()
    # mse = custom_loss
    mse = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = Lion(model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    import random

    # 训练模型
    model.train()
    years = list(range(2010, 2020))
    random.shuffle(years)
    for epoch in range(epochs):
        for year in years:

            running_loss = 0.0
            true_loss = 0.0
            model = model.to(device)

            wind_u, wind_v, wave_height, _, _ = data_preprocess(str(year))

            # 输出数据集和标签的形状
            print(f"Epoch:{epoch + 1},Year:{year}")

            dataset = DynamicDataset(wind_u, wind_v, wave_height, time)
            dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)

            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                target_size = (lat, lon)
                x_u = F.interpolate(data[:, 0, :], size=target_size,
                                    mode='bilinear', align_corners=False)
                x_v = F.interpolate(data[:, 1, :], size=target_size,
                                    mode='bilinear', align_corners=False)
                outputs = model(x_u, x_v)  # torch.Size([20, 72, 14, 32])
                outputs = torch.masked_fill(outputs, mask, 0)
                loss = mse(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # mask = (labels != 0).float()
                true_loss += (torch.sum((outputs - labels) ** 2)) / torch.sum(mask)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Rmse Loss: {np.sqrt(running_loss / len(dataloader))},"
                  f"True Loss:{true_loss / len(dataloader)}, "
                  f"Current Learning Rate: {current_lr}")

            writer.add_scalar('Learning Rate:', current_lr, global_step=epoch)
            writer.add_scalar('Train Rmse Loss:', np.sqrt(running_loss / len(dataloader))
                              , global_step=epoch)
            if epoch % 10 == 0:
                writer.add_image('predict', outputs[0, :].unsqueeze(0), global_step=epoch)
                writer.add_image('label', labels[0, :].unsqueeze(0), global_step=epoch)

                rmse = torch.sqrt(torch.mean(torch.square(outputs - labels), 0)) \
                    .detach().cpu().numpy()

                # 创建Matplotlib图表并绘制RMSE
                plt.figure()
                vmin = np.min(rmse[np.nonzero(rmse)])
                vmax = np.percentile(rmse, 99)

                cax = plt.matshow(rmse, cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(cax)


                writer.add_figure('RMSE Heatmap', plt.gcf(), global_step=epoch)
                plt.close(plt.gcf())

            del data, labels

        validation_loss = 0
        real_loss = 0
        validation_step = 0
        rmse = np.zeros([lat, lon])

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():

                for validation_data, target in validation_dataloader:
                    validation_data, target = validation_data.to(device), target.to(device)

                    target_size = (lat, lon)

                    x_u = F.interpolate(validation_data[:, 0, :], size=target_size,
                                        mode='bilinear', align_corners=False)
                    x_v = F.interpolate(validation_data[:, 1, :], size=target_size,
                                        mode='bilinear', align_corners=False)

                    logits = model(x_u, x_v)
                    logits = torch.masked_fill(logits, mask, 0)

                    msevalue = mse(logits, target).item()

                    realvalue = torch.mean(torch.abs(logits - target))
                    validation_loss += msevalue
                    real_loss += realvalue
                    rmse += torch.sqrt(torch.mean(torch.square(logits - target), 0)) \
                        .detach().cpu().numpy()

                    validation_step += 1

                    if validation_step == 100:
                        break

                validation_loss /= validation_step
                real_loss /= validation_step
                rmse /= validation_step

                print(
                    '\n  Epoch: {} validation set: Average Mse loss: {:.6f},'
                    'Average RMse loss: {:.6f}, Abs loss: {:.6f}'
                    .format(epoch, validation_loss, np.sqrt(validation_loss), real_loss))
                writer.add_scalar('Validation RMSE Loss:', np.sqrt(validation_loss), global_step=epoch)

                plt.figure()
                vmin = np.min(rmse[np.nonzero(rmse)])
                vmax = np.percentile(rmse, 99)

                cax = plt.matshow(rmse, cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(cax)
                # 将Matplotlib图表添加到TensorBoard
                writer.add_figure('RMSE validation Heatmap', plt.gcf(), global_step=epoch)
                plt.close(plt.gcf())

        if epoch % 10 == 0 and epoch > 1:
            save_dict = {}
            save_dict[f'epoch'] = epoch
            save_dict[f'model_state_dict'] = model.state_dict()
            save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
            save_dict[f'loss'] = np.sqrt(validation_loss)

            checkpoint_path = os.path.join(model_path,
                                "{}_{}_mid_".format(epoch,
                                (running_loss / len(dataloader))) + info + ".pt")
            torch.save(save_dict, checkpoint_path)
            print("保存点")

    save_dict = {}
    save_dict[f'epoch'] = epoch
    save_dict[f'model_state_dict'] = model.state_dict()
    save_dict[f'optimizer_state_dict'] = optimizer.state_dict()
    save_dict[f'loss'] = np.sqrt(validation_loss)

    checkpoint_path = os.path.join(model_path, "{}_{}.pt".
                                   format(epoch, (running_loss / len(dataloader))))
    torch.save(save_dict, checkpoint_path)
    print("训练完成")


def continuous_inference(model, test_dataloader,biascorrection=False):
    total_time = int(len(test_dataloader))

    rmse_figure_data = np.zeros([2, total_time, lat, lon])
    out_data = []
    label_data = []

    count = 0

    if biascorrection == True:
        lmodel = np.load("model/blacksea_unet_linearregression.npz")
        k = torch.reshape(torch.tensor(lmodel['k']), [1, lat, lon]).to(device)
        b = torch.reshape(torch.tensor(lmodel['b']), [1, lat, lon]).to(device)


    with torch.no_grad():
        for test_data, label  in test_dataloader:
            test_data, label  = test_data.to(device), label .to(device)
            target_size = (lat, lon)

            x_u = F.interpolate(test_data[:, 0, :], size=target_size,
                                mode='bilinear', align_corners=False)

            x_v = F.interpolate(test_data[:, 1, :], size=target_size,
                                mode='bilinear', align_corners=False)
            out = model(x_u, x_v)
            out = torch.unsqueeze(out, 0)

            out[label  == 0] = 0



            if biascorrection == True:
                out = out * k + b


            loss = torch.mean(torch.square(out - label))

            varnumber = 0
            out_data.append(
                np.squeeze(np.reshape(out.detach().cpu().numpy(), [lat, lon])))
            label_data.append(np.squeeze(np.reshape(label.detach().cpu().numpy(), [lat, lon])))

            rmse_figure_data[0, count, :, :] = \
                np.reshape(out.detach().cpu().numpy(), [lat, lon])  # predict:0

            rmse_figure_data[1, count, :, :] = \
                np.reshape(label.detach().cpu().numpy(), [lat, lon])  # label:1

            print(' from {} hours inference {} step Mse Loss: {:.6f}  '.format(count, 1, loss))

            count += 1
            if count == total_time: break

    rmse_figure = np.sqrt(np.mean(np.square(rmse_figure_data[0, :] - rmse_figure_data[1, :]), 0))


    relative_rmse_figure = np.zeros([lat, lon])
    label_mean = np.mean(label_data, 0)
    relative_rmse_figure = rmse_figure / np.where(label_mean == 0, 1, label_mean)


    correlation_coefficients = np.zeros([lat, lon])
    bias = np.zeros([lat, lon])

    bias = np.mean(rmse_figure_data[0, :] - rmse_figure_data[1, :], axis=0)

    # 遍历每个地理位置
    for i in range(rmse_figure_data.shape[2]):
        for j in range(rmse_figure_data.shape[3]):
            # 提取该位置的预测值和真实值
            predictions = rmse_figure_data[0, :, i, j]
            actuals = rmse_figure_data[1, :, i, j]
            if np.all(actuals == 0) == True:
                correlation_coefficients[i, j] = 1
                continue

            # 计算相关系数
            correlation = np.corrcoef(predictions, actuals)[0, 1]

            # 将相关系数存储在数组中
            correlation_coefficients[i, j] = correlation

    return rmse_figure, relative_rmse_figure, correlation_coefficients, bias, out_data, label_data


def NetInference():

    checkpoint = torch.load(model_path+'blacksea_unet.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # model=torch.load(model_path+'black_sea_unet.pt')

    model.eval()
    for year in range(2020, 2021):

        wind_u, wind_v, wave_height, axis_lat, axis_lon = data_preprocess(str(year))
        dataset = DynamicDataset(wind_u, wind_v, wave_height, time)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        rmse_figure_data, rrmse_figure_data, correlation_coefficients, \
            bias, out_data, label_data = continuous_inference(model, dataloader,True)

        # np.savez('plot_' + info + '_' + str(year) + '.npz',
        #          rmse_figure_data=rmse_figure_data,
        #          rrmse_figure_data=rrmse_figure_data,
        #          correlation_coefficients=correlation_coefficients,
        #          bias=bias,
        #          out_data=out_data,
        #          label_data=label_data,
        #          axis_lat=axis_lat,
        #          axis_lon=axis_lon)
        #
        del rmse_figure_data, rrmse_figure_data, correlation_coefficients, \
            out_data, label_data, wind_u, wind_v, wave_height, axis_lat, axis_lon


def BiasCorrection():
    checkpoint = torch.load(r'.\model\blacksea_72h.pt')
    model.load_state_dict(checkpoint['model_state_dict'])



    model.eval()
    for year in range(2010, 2020):#just use the training set
        print(year)
        wind_u, wind_v, wave_height, axis_lat, axis_lon = data_preprocess(str(year))
        dataset = DynamicDataset(wind_u, wind_v, wave_height, time)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        lat = axis_lat.shape[0]
        lon = axis_lon.shape[0]

        rmse_figure_data, rrmse_figure_data, correlation_coefficients \
            , bias, out_data, label_data = continuous_inference(model, dataloader)

        k_values = np.ones((lat, lon))
        b_values = np.zeros((lat, lon))
        out_data = np.array(out_data)
        label_data = np.array(label_data)

        for i in tqdm(range(lat)):
            for j in range(lon):

                if (label_data[:, i, j] != 0).all():

                    modell = LinearRegression().fit(out_data[:, i, j].reshape(-1, 1),
                                                    label_data[:, i, j].reshape(-1, 1))
                    # 存储k和b
                    k_values[i, j] = modell.coef_[0]
                    b_values[i, j] = modell.intercept_

        #Generate an npz file of linear regression k b values for each year,
        # which should be averaged before being applied to the model
        np.savez("BlackSea_Linear/liner_train_" + str(year) + ".npz", k=k_values, b=b_values)
        del rmse_figure_data, rrmse_figure_data, correlation_coefficients, \
            out_data, label_data, wind_u, wind_v, wave_height, axis_lat, axis_lon


#SWH Range
lat = 131
lon = 296

#Wind Range
lat_wind = 27
lon_wind = 60

time = 72

model_path = r"./model/"
if not os.path.exists(model_path):
    os.makedirs(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet(n_vars=2, n_times=time, n_classes=1)
model = model.to(device)


info = "BlackSea_"

#NetTrain()
#BiasCorrection()
NetInference()
