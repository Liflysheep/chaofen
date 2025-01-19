import numpy as np
import scipy.io as sio
import collections
import xlrd
import os
import scipy

def loadData(mname):
    return sio.loadmat(mname)

def get_spectral_response(data_name, sp_root_path='./spectral_response'):
    """
    从 .xls 文件中提取光谱响应函数（SRF）。
    :param data_name: 数据名称（用于确定 .xls 文件路径）。
    :param sp_root_path: 光谱响应文件的根目录。
    :return: 归一化后的光谱响应矩阵。
    """
    if 'toy' in data_name:
        xls_path = os.path.join(sp_root_path, 'toys.xls')
    elif data_name == 'Chikusei':
        xls_path = os.path.join(sp_root_path, 'TG.xls')
    else:
        xls_path = os.path.join(sp_root_path, data_name + '.xls')

    if not os.path.exists(xls_path):
        raise Exception("Spectral response path does not exist: %s" % xls_path)

    # 读取 .xls 文件
    data = xlrd.open_workbook(xls_path)
    table = data.sheets()[0]
    num_cols = table.ncols

    # 提取每一列数据并拼接成矩阵，跳过第一列
    cols_list = [np.array(table.col_values(i)).reshape(-1, 1) for i in range(num_cols)]  # 从第2列开始
    sp_data = np.concatenate(cols_list, axis=1).T  # 不转置，直接按列拼接
    # print(sp_data.shape)

    # 归一化光谱响应
    sp_data = sp_data / (sp_data.sum(axis=0))
    return sp_data.astype(np.float32)

def readData(filename, num=10, sp_root_path='./spectral_response',toy_scale=16):
    """
    读取数据并处理。
    :param filename: 数据文件路径。
    :param num: 数据编号（默认为 10）。
    :param sp_root_path: 光谱响应文件的根目录。
    :return: 包含处理后的数据的命名元组。
    """
    input = loadData(filename)
    data = collections.namedtuple('data', [
        'hyperLR', 'multiHR', 'hyperHR', 'dimLR', 'dimHR', 'srf', 'srfactor',
        'colLR', 'meanLR', 'reducedLR', 'sphere', 'num'
    ], verbose=False)

    # 根据文件名动态调整索引
    if 'toy' in filename:
        data.multiHR = np.array(input['RGB']).astype(np.float32)
        if toy_scale==16:
            data.hyperLR = np.array(input['LR_16']).astype(np.float32)
        elif toy_scale==32:
            data.hyperLR = np.array(input['LR_32']).astype(np.float32)
        else:
            data.hyperHR = np.array(input[''])
        data.hyperHR = np.array(input['HR']).astype(np.float32)
    elif 'Chikusei' in filename:
        data.hyperLR = np.array(input['hsi']).astype(np.float32)
        data.multiHR = np.array(input['msi']).astype(np.float32)
        data.hyperHR = np.array(input['HR']).astype(np.float32)
    elif 'AISA' in filename:
        data.hyperLR = np.array(input['hsi']).astype(np.float32)
        data.multiHR = np.array(input['msi']).astype(np.float32)
        data.hyperHR = np.array(input['HR']).astype(np.float32)

    data.dimLR = data.hyperLR.shape
    data.dimHR = data.multiHR.shape
    data.num = num

    # 从 .xls 文件中提取光谱响应函数
    data_name = os.path.splitext(os.path.basename(filename))[0]  # 从文件名中提取数据名称
    data.srf = get_spectral_response(data_name, sp_root_path)

    # 计算超分辨率因子
    data.srfactor = np.divide(data.dimHR[0], data.dimLR[0]).astype(np.int)

    #计算初始插值HR
    # data.hyperLRI = np.array(input['hyperLRI']).astype(np.float32)
    from scipy.ndimage import zoom
    hyperLR = data.hyperLR
    # 双三次插值
    data.hyperLRI = zoom(hyperLR, zoom=(data.srfactor, data.srfactor, 1), order=3)

    # 处理低分辨率高光谱图像
    data.col_lr_hsi = np.reshape(data.hyperLR, [data.dimLR[0] * data.dimLR[1], data.dimLR[2]])
    data.mean_lr_hsi = np.mean(data.col_lr_hsi, axis=0, keepdims=True)
    data.rcol_lr_hsi = np.subtract(data.col_lr_hsi, data.mean_lr_hsi)
    data.img_lr_hsi = np.reshape(data.rcol_lr_hsi, [data.dimLR[0], data.dimLR[1], data.dimLR[2]])

    # 处理高分辨率多光谱图像
    data.col_hr_msi = np.reshape(data.multiHR, [data.dimHR[0] * data.dimHR[1], data.dimHR[2]])
    data.mean_hr_msi = np.mean(data.col_hr_msi, axis=0, keepdims=True)
    data.rcol_hr_msi = np.subtract(data.col_hr_msi, data.mean_hr_msi)
    data.img_hr_msi = np.reshape(data.rcol_hr_msi, [data.dimHR[0], data.dimHR[1], data.dimHR[2]])

    # 生成低分辨率多光谱图像
    data.multiLR = scipy.ndimage.zoom(data.multiHR, zoom=[1.0 / data.srfactor, 1.0 / data.srfactor, 1], order=0)
    data.col_lr_msi = np.reshape(data.multiLR, [data.dimLR[0] * data.dimLR[1], data.dimHR[2]])
    data.mean_lr_msi = np.mean(data.col_lr_msi, axis=0, keepdims=True)
    data.rcol_lr_msi = np.subtract(data.col_lr_msi, data.mean_lr_msi)

    # 处理高分辨率高光谱图像
    data.col_hr_hsi = np.reshape(data.hyperHR, [data.dimHR[0] * data.dimHR[1], data.dimLR[2]])

    return data