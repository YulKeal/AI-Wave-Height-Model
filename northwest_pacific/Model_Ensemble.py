import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import os


lat=91
lon=71



def train():
    k = []
    b = []

    for year in range(2000,2018):

        data1=np.load('EF Path'+str(year)+'.npz')
        data2=np.load('Unet Path'+str(year)+'.npz')



        print("year:{}".format(year))

        #
        # if (data1['label_data']==data2['label_data']).all():
        #     print("year:{} pass".format(year))

        label_data=data1['label_data']

        EF=data1['out_data']

        axis_lat = data1['axis_lat']
        axis_lon = data1['axis_lon']
        lat=axis_lat.shape[0]
        lon=axis_lon.shape[0]

        UNet=data2['out_data']
       # UNet[:,mask==False]= 0

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



    np.savez("./results/modelensemble_extend.npz", k=kmean, b=bmean)



def get_state(out_data,label_data):


    rmse_figure_data=np.stack((out_data, label_data), axis=0)


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

         
            correlation = np.corrcoef(predictions, actuals)[0, 1]

         
            correlation_coefficients[i, j] = correlation

    return rmse_figure, relative_rmse_figure, correlation_coefficients, bias


def test(year=2020):
    data1 = np.load('EF Path' + str(year) + '.npz')
    data2 = np.load('Unet Path' + str(year) + '.npz')

    lmodel=np.load("./results/modelensemble_extend.npz")
    k=lmodel['k']
    b=lmodel['b']


    print("year:{}".format(year))



    label_data = data1['label_data']

    EF = data1['out_data']

    axis_lat = data1['axis_lat']
    axis_lon = data1['axis_lon']
    lat = axis_lat.shape[0]
    lon = axis_lon.shape[0]

    UNet = data2['out_data']

    EF=np.array(EF,dtype=np.float32)
    UNet=np.array(UNet,dtype=np.float32)
    label_data=np.array(label_data,dtype=np.float32)
    k = np.array(k, dtype=np.float32)
    b = np.array(b, dtype=np.float32)


    out_data=EF * k[:,:,0] +UNet *k[:,:,1] +b
    del EF,UNet,k,b


    print("Correction completed.")

    rmse_figure_data, rrmse_figure_data, correlation_coefficients, bias=get_state(out_data,label_data)




    rmse_figure_data[rmse_figure_data == 0] = np.nan
    correlation_coefficients[correlation_coefficients == 1] = np.nan
    rrmse_figure_data[rrmse_figure_data == 0] = np.nan
    bias[ bias == 0] = np.nan
      
    results_path = '/model-ensemble-extend/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    plot_path = '/model-ensemble-extend/plot/'
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


#train()
test(2020)