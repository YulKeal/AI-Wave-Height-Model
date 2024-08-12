import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import os
import plotfig
lat=281
lon=720



def train():
    k = []
    b = []

    for year in range(2000,2018):

        data1=np.load('D:/EartherFormerResults/epochensemble/plot_era5_global_HR0.5_4ptensemble_'+str(year)+'.npz')
        data2=np.load('D:/EartherFormerResults/Unet_Results/plot_era5_unet_global__'+str(year)+'.npz')

        mask=np.load('mask_global_HRSWH.npz')['mask']

        print("year:{}".format(year))

        #
        # if (data1['label_data']==data2['label_data']).all():
        #     print("year:{} pass".format(year))
        # else:
        #     print("Error")

        label_data=data1['label_data']

        EF=data1['out_data']

        axis_lat = data1['axis_lat']
        axis_lon = data1['axis_lon']
        lat=axis_lat.shape[0]
        lon=axis_lon.shape[0]

        UNet=data2['out_data']
        UNet[:,mask==False]= 0

        k_values = np.ones((lat, lon,2))
        b_values = np.zeros((lat, lon))




        for i in tqdm(range(lat)):
            for j in range(lon):
                if ((label_data[:, i, j] != 0).all()):

                    X = np.concatenate((EF[:, i, j].reshape(-1, 1), UNet[:, i, j].reshape(-1, 1)), axis=1)
                  #  print(X.shape)
                    Y = label_data[:, i, j].reshape(-1, 1)

                    modell = LinearRegression().fit(X, Y)

                    k_values[i, j,0] = modell.coef_[0][0]
                    k_values[i, j, 1] = modell.coef_[0][1]
                    b_values[i, j] = modell.intercept_[0]
                #    print("k1={},k2={},b={}".format(modell.coef_[0][0],modell.coef_[0][1],modell.intercept_[0]))

        k.append(k_values)
        b.append(b_values)


    k=np.array(k)
    b=np.array(b)


    kmean=np.mean(k,axis=0)
    bmean=np.mean(b,axis=0)

    np.savez("liner_train_modelensemble.npz", k=kmean, b=bmean)



def get_state(out_data,label_data):


    rmse_figure_data=np.stack((out_data, label_data), axis=0)


    rmse_figure = np.sqrt(np.mean(np.square(rmse_figure_data[0, :] - rmse_figure_data[1, :]), 0))
    # np.savez('predict_'+ps+'.npz',
    #          predict=rmse_figure_data[0, :, :, :])

    relative_rmse_figure = np.zeros([lat, lon])
    label_mean = np.mean(label_data, 0)
    relative_rmse_figure = rmse_figure / np.where(label_mean == 0, 1, label_mean)

    # 初始化一个与纬度和经度维度相同的数组，用于存储相关系数
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

    return rmse_figure, relative_rmse_figure, correlation_coefficients, bias


def test(year=2020):

    data1 = np.load('D:/EartherFormerResults/epochensemble/plot_era5_global_HR0.5_4ptensemble_' + str(year) + '.npz')
    data2 = np.load('D:/EartherFormerResults/Unet_Results/plot_era5_unet_global__' + str(year) + '.npz')

    mask = np.load('mask_global_HRSWH.npz')['mask']
    lmodel=np.load("liner_train_modelensemble.npz")
    k=lmodel['k']
    b=lmodel['b']


    print("year:{}".format(year))

    #
    # if (data1['label_data']==data2['label_data']).all():
    #     print("year:{} pass".format(year))
    # else:
    #     print("Error")

    label_data = data1['label_data']

    EF = data1['out_data']

    axis_lat = data1['axis_lat']
    axis_lon = data1['axis_lon']
    lat = axis_lat.shape[0]
    lon = axis_lon.shape[0]

    UNet = data2['out_data']
    UNet[:, mask == False] = 0


    out_data=EF * k[:,:,0] +UNet *k[:,:,1] +b
    #out_data=EF
    rmse_figure_data, rrmse_figure_data, correlation_coefficients, bias=get_state(out_data,label_data)


    rmse_figure_data[rmse_figure_data == 0] = np.nan
    correlation_coefficients[correlation_coefficients == 1] = np.nan
    rrmse_figure_data[rrmse_figure_data == 0] = np.nan
    bias[ bias == 0] = np.nan
        #
    results_path = 'D:/EartherFormerResults/epochensemble/testset/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    plot_path = 'D:/EartherFormerResults/epochensemble/testset/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    np.savez(results_path + 'plot_modelensemble'  + '_' + str(year) + '.npz',
             rmse_figure_data=rmse_figure_data,
             rrmse_figure_data=rrmse_figure_data,
             correlation_coefficients=correlation_coefficients,
             bias=bias,
             out_data=out_data,
             label_data=label_data,
             axis_lat=axis_lat,
             axis_lon=axis_lon)

    plotfig.Plot_Scatter(out_data, label_data, time_step=None, filename=plot_path+str(year) + 'global_sc')
    plotfig.Plot_RMSE_N(rmse_figure_data, axis_lat, axis_lon, filename=plot_path+str(year) + 'global_rmse',cbname='RMSE (m)')
    plotfig.Plot_RMSE_N(correlation_coefficients, axis_lat, axis_lon, filename=plot_path+str(year) + 'global_cor',
                         figurename='Spatial Correlation Map',cbname='Correlation Coefficients')
    plotfig.Plot_RMSE_N(rrmse_figure_data, axis_lat, axis_lon,
                            filename=plot_path+str(year) + '_rrmse', figurename='Relative RMSE Spatial Distribution Map',cbname='Relative RMSE')
    plotfig.Plot_RMSE_N(bias, axis_lat, axis_lon,
                            filename=plot_path+str(year) + '_bias', figurename='Bias Spatial Distribution Map',cbname='Bias (m)')

#train()
# for year in range(2021,2023):
#    test(year)
test(2019)
