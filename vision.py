# coding=utf-8
import cv2
import random
import numpy as np
from scipy.io import loadmat  # mat文件读取
from scipy.io import savemat  # mat文件保存
import time
from multiprocessing import Pool

import HaveFun.common as common


def randomSample(img, block_size, num_sample):
    """
    在一个影像中随机采样指定大小、数量的影像块 -> blocks, indices

    :param img: 输入影像
    :param block_size: 采样影像块大小
    :param num_sample: 采样影像块个数
    :return: 采样得到的影像块、索引
    """
    blocks = []
    indices = []
    counter = 0

    img_width = img.shape[1]
    img_height = img.shape[0]

    while counter < num_sample:
        tmp_x_start = random.randint(0, img_width - block_size)
        tmp_y_start = random.randint(0, img_height - block_size)
        tmp_x_end = tmp_x_start + block_size
        tmp_y_end = tmp_y_start + block_size

        block = img[tmp_y_start:tmp_y_end, tmp_x_start:tmp_x_end]

        blocks.append(block)
        indices.append([tmp_x_start, tmp_x_end, tmp_y_start, tmp_y_end, 0])
        counter += 1
    return blocks, indices


def randomSamplePair(img_in, img_gt, block_size, num_sample):
    """
    在输入影像和真值影像中随机采样指定大小、数量的配对影像块 -> blocks_in, blocks_gt, indices
    :param img_in: 输入的影像
    :param img_gt: 输入的真值影像
    :param block_size: 影像块大小
    :param num_sample: 影像块个数
    :return: 采样后的匹配输入影像块、真值影像块和索引
    """
    blocks_in = []
    blocks_gt = []
    indices = []
    counter = 0

    img_width = img_in.shape[1]
    img_height = img_in.shape[0]

    while counter < num_sample:
        tmp_x_start = random.randint(0, img_width - block_size)
        tmp_y_start = random.randint(0, img_height - block_size)
        tmp_x_end = tmp_x_start + block_size
        tmp_y_end = tmp_y_start + block_size

        block_in = img_in[tmp_y_start:tmp_y_end, tmp_x_start:tmp_x_end]
        block_gt = img_gt[tmp_y_start:tmp_y_end, tmp_x_start:tmp_x_end]

        blocks_in.append(block_in)
        blocks_gt.append(block_gt)
        indices.append([tmp_x_start, tmp_x_end, tmp_y_start, tmp_y_end, 0])
        counter += 1
    return blocks_in, blocks_gt, indices


def enhanceSamples(in_blocks, indices):
    """
    对影像块进行增强(水平翻转、竖直翻转、水平+竖直翻转) -> enhanced_blocks, enhanced_indices
    :param in_blocks: 待增强的影像块
    :param indices: 待增强影像块对应的索引
    :return: 增强后影像块以及对应的索引
    """
    enhanced_blocks = []
    enhanced_indices = []
    for i in range(len(in_blocks)):
        tmp_block = in_blocks[i]

        tmp_block_1 = cv2.flip(tmp_block, 1)  # 水平翻转
        tmp_block_2 = cv2.flip(tmp_block, 0)  # 竖直翻转
        tmp_block_3 = cv2.flip(tmp_block_2, 0)  # 竖直+水平翻转

        enhanced_blocks.append(tmp_block)
        enhanced_blocks.append(tmp_block_1)
        enhanced_blocks.append(tmp_block_2)
        enhanced_blocks.append(tmp_block_3)

        enhanced_indices.append(indices[i])
        enhanced_indices.append([indices[i][0], indices[i][1], indices[i][2], indices[i][3], 1])
        enhanced_indices.append([indices[i][0], indices[i][1], indices[i][2], indices[i][3], 2])
        enhanced_indices.append([indices[i][0], indices[i][1], indices[i][2], indices[i][3], 3])
    return enhanced_blocks, enhanced_indices


def enhanceSamplesPair(in_blocks, gt_blocks, indices):
    """
    对成对的输入影像块、真值影像块进行增强(水平翻转、竖直翻转、水平+竖直翻转) -> enhanced_in_blocks, enhanced_gt_blocks, enhanced_indices
    :param in_blocks: 输入影像块
    :param gt_blocks: 真值影像块
    :param indices: 对应的影像块索引
    :return: 增强以后的配对影像块和真值块以及索引
    """
    enhanced_in_blocks = []
    enhanced_gt_blocks = []
    enhanced_indices = []
    for i in range(len(in_blocks)):
        tmp_in_block = in_blocks[i]
        tmp_gt_block = gt_blocks[i]

        tmp_in_block_1 = cv2.flip(tmp_in_block, 1)  # 水平翻转
        tmp_gt_block_1 = cv2.flip(tmp_gt_block, 1)

        tmp_in_block_2 = cv2.flip(tmp_in_block, 0)  # 竖直翻转
        tmp_gt_block_2 = cv2.flip(tmp_gt_block, 0)

        tmp_in_block_3 = cv2.flip(tmp_in_block_2, 0)  # 竖直+水平翻转
        tmp_gt_block_3 = cv2.flip(tmp_gt_block_2, 0)

        enhanced_in_blocks.append(tmp_in_block)
        enhanced_in_blocks.append(tmp_in_block_1)
        enhanced_in_blocks.append(tmp_in_block_2)
        enhanced_in_blocks.append(tmp_in_block_3)

        enhanced_gt_blocks.append(tmp_gt_block)
        enhanced_gt_blocks.append(tmp_gt_block_1)
        enhanced_gt_blocks.append(tmp_gt_block_2)
        enhanced_gt_blocks.append(tmp_gt_block_3)

        enhanced_indices.append(indices[i])
        enhanced_indices.append([indices[i][0], indices[i][1], indices[i][2], indices[i][3], 1])
        enhanced_indices.append([indices[i][0], indices[i][1], indices[i][2], indices[i][3], 2])
        enhanced_indices.append([indices[i][0], indices[i][1], indices[i][2], indices[i][3], 3])
    return enhanced_in_blocks, enhanced_gt_blocks, enhanced_indices


def outputSamples(out_dir, out_type, img_name, color_mode, in_blocks, gt_blocks, indices, vali_rate):
    """
    将采样好的影像块分割为训练与验证部分，并分别输出到文件
    :param out_dir: 输出的根目录
    :param out_type: 输出的影响块的文件类型
    :param img_name: 输出影像块文件的前缀
    :param color_mode: 影像文件块的彩色顺序
    :param in_blocks: 输入的影像块列表
    :param gt_blocks: 输入的真值块列表
    :param indices: 输入的对应索引列表
    :param vali_rate: 影像块中验证集的比例
    :return: 无返回值
    """
    train_input_dir = out_dir + "/train/input"
    train_groundtruth_dir = out_dir + "/train/groundtruth"
    common.isDirExist(train_input_dir)
    common.isDirExist(train_groundtruth_dir)

    vali_input_dir = out_dir + "/validation/input"
    vali_groundtruth_dir = out_dir + "/validation/groundtruth"
    common.isDirExist(vali_input_dir)
    common.isDirExist(vali_groundtruth_dir)

    index_dir = out_dir + "/indices"
    common.isDirExist(index_dir)

    if out_type[0] != ".":
        out_type = "." + out_type

    vali_num = int(vali_rate * len(in_blocks)) + 1
    vali_step = int(len(in_blocks) / vali_num)

    fout = open(index_dir + "/" + img_name.split(".")[0] + "_indices.txt", "w")
    fout.write("# number\tx_start\tx_end\ty_start\ty_end\ttype\tpattern\n")
    for i in range(len(in_blocks)):
        if i % vali_step == 0:
            tmp_gt_path = vali_groundtruth_dir
            tmp_in_path = vali_input_dir
            tmp_flag = 'v'
        else:
            tmp_gt_path = train_groundtruth_dir
            tmp_in_path = train_input_dir
            tmp_flag = 't'

        # 如果彩色通道顺序是RGB，就转换一下，否则不用额外操作
        if color_mode.__contains__("rgb") or color_mode.__contains__("RGB") or color_mode.__contains__("Rgb"):
            tmp_raw = cv2.cvtColor(in_blocks[i], cv2.COLOR_RGB2BGR)
            tmp_gt = cv2.cvtColor(gt_blocks[i], cv2.COLOR_RGB2BGR)
        else:
            tmp_raw = in_blocks[i]
            tmp_gt = gt_blocks[i]

        cv2.imwrite(tmp_in_path + "/" + img_name.split(".")[0] + "_" + i.__str__().zfill(5) + "_input" + out_type,
                    tmp_raw)
        cv2.imwrite(tmp_gt_path + "/" + img_name.split(".")[0] + "_" + i.__str__().zfill(5) + "_gt" + out_type, tmp_gt)
        fout.write(i.__str__().zfill(5) + "\t" +
                   str(indices[i][0]) + "\t" + str(indices[i][1]) + "\t" +
                   str(indices[i][2]) + "\t" + str(indices[i][3]) + "\t" +
                   tmp_flag + "\t" +
                   str(indices[i][4]) + "\n")
    fout.close()


def drawBlocks(in_img, indices, color=(128, 200, 180)):
    """
    绘制影像块范围 -> img_rgb
    :param in_img: 待绘制的影像底图
    :param indices: 待绘制的影像块索引
    :param color: 影像块绘制颜色
    :return: 绘制好的影像
    """
    img = np.zeros([in_img.shape[0], in_img.shape[1]], np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(indices)):
        start_x = indices[i][0]
        end_x = indices[i][1]
        start_y = indices[i][2]
        end_y = indices[i][3]
        cv2.rectangle(img_rgb, (start_x, start_y), (end_x, end_y), color, thickness=2)
    return img_rgb


def getBlockRange(img, row=2, col=2):
    """
    将较大影像分块(指定个数)，返回每块的坐标索引 -> blocks
    :param img: 原始影像
    :param row: 待分的行数，默认为2
    :param col: 待分的列数，默认为2
    :return: 包含每块坐标索引的列表
    """

    img_h = img.shape[1]
    img_w = img.shape[0]
    # print img_h, img_w
    w_per_block = img_w / row
    h_per_block = img_h / col
    # print h_per_block, w_per_block
    blocks = []
    for i in range(row):
        for j in range(col):
            w = i * w_per_block
            h = j * h_per_block
            rb_w = w + w_per_block
            rb_h = h + h_per_block
            # print w, '-', rb_w, h, '-', rb_h
            blocks.append([w, rb_w, h, rb_h])
    return blocks


def calcMSE(img1, img2):
    """
    计算两个影像(灰度)的MSE灰度差异 -> mse
    :param img1: 输入的影像1
    :param img2: 输入的影像2
    :return: 计算的MSE
    """
    band1 = img1.astype(np.float)
    band2 = img2.astype(np.float)

    diff = np.power(band1 - band2, 2)
    mse = np.sum(diff) / (band1.shape[0] * band1.shape[1])
    return mse


def evaluateMSE(gt_img, input_img):
    """
    calcMSE()函数的进一步封装，适用于彩色或灰度图像计算MSE -> mse
    :param gt_img: 输入的真值影像
    :param input_img: 输入的测试影像
    :return: 计算的MSE
    """

    # for rgb color images
    if len(input_img.shape) == 3:
        in_band_r = input_img[:, :, 0]
        in_band_g = input_img[:, :, 1]
        in_band_b = input_img[:, :, 2]
        gt_band_r = gt_img[:, :, 0]
        gt_band_g = gt_img[:, :, 1]
        gt_band_b = gt_img[:, :, 2]

        mse_r = calcMSE(gt_band_r, in_band_r)
        mse_g = calcMSE(gt_band_g, in_band_g)
        mse_b = calcMSE(gt_band_b, in_band_b)
        mse_mean = (mse_r + mse_g + mse_b) / 3
        return mse_mean
    # for grayscale(single band) images
    else:
        mse = calcMSE(gt_img, input_img)
        return mse


def evaluatePSNR(gt_img, input_img, bit_level=8):
    """
    计算两个影像间的峰值信噪比(默认为8bit) -> psnr
    :param gt_img: 输入的真值影像
    :param input_img: 输入的测试影像
    :param bit_level: 灰度级数
    :return: 计算的psnr
    """
    mean_mse = evaluateMSE(gt_img, input_img)
    if mean_mse == 0:
        mean_mse = 0.000000001
    psnr = 10 * np.log10((np.power(2, 8) - 1) ** 2 / mean_mse)
    return psnr


def cvtImgs2Mat(img_dir, file_type, img_key_name):
    """
    将多个影像文件合并成一个Matlab的mat对象 -> img_dict
    :param img_dir: 寻找影像的文件夹
    :param file_type: 寻找的影像类型
    :param img_key_name: 输出mat文件中数据的key名称
    :return: 合并好的mat对象
    """
    paths, names, files = common.findFiles(img_dir, file_type)

    imgs = []
    for i in range(len(files)):
        tmp_img = cv2.imread(files[i])
        imgs.append(tmp_img)

    img_width = imgs[0].shape[1]
    img_height = imgs[0].shape[0]
    num_channel = imgs[0].shape[2]
    num_imgs = len(imgs)
    img_mat = np.zeros([num_imgs, img_height, img_width, num_channel], np.uint8)

    for i in range(len(imgs)):
        img_mat[i, :, :, :] = imgs[i]

    img_dict = {img_key_name: img_mat,
                '__header__': 'Matlab MAT-file, Created by Xuhui Zhao on ' + time.ctime(),
                '__version__': '1.0',
                '__globals__': ''}
    return img_dict


def cvtImgs2MatAndSave(img_dir, file_type, img_key_name, out_path):
    """
    将多个影像文件合并成一个Matlab的mat文件并输出
    :param img_dir: 寻找影像的文件夹
    :param file_type: 寻找的影像类型
    :param img_key_name: 输出mat文件中数据的key名称
    :param out_path: 输出mat文件的路径
    :return: 无返回值
    """
    img_dict = cvtImgs2Mat(img_dir, file_type, img_key_name)
    savemat(out_path, img_dict)


def cvtMat2Imgs(mat_path, img_key_name):
    """
    将Matlab的mat类型文件转换成图片 -> imgs
    :param mat_path: 输入的Matlab的mat文件路径
    :param img_key_name: mat文件中影像的key名称
    :return: 解析好的影像
    """
    imgs = []
    img_mat = loadmat(mat_path)
    img_data = img_mat[img_key_name]
    for i in range(img_data.shape[0]):
        imgs.append(img_data[i, :, :, :])
    return imgs


def cvtMat2ImgsAndSave(mat_path, img_key_name, img_out_dir, img_type):
    """
    将Matlab的mat类型文件转换成图片并输出
    :param mat_path: 输入的Matlab的mat文件路径
    :param img_key_name: mat文件中影像的key名称
    :param img_out_dir: 输出影像的文件夹
    :param img_type: 输出影像的文件类型
    :return: 无
    """
    imgs = cvtMat2Imgs(mat_path, img_key_name)

    for i in range(len(imgs)):
        cv2.imwrite(img_out_dir + "/" + str(i).zfill(5) + "." + img_type, imgs[i])


def cropImage(img, block_height, block_width):
    """
    将输入影像划分成block_height×block_width大小的影像块(所有影像块大小相同，不足部分扩边得到) -> img_blocks, block_indices, block_param
    :param img: 输入影像
    :param block_height: 影像块高度
    :param block_width: 影像块宽度
    :return: 裁剪后的影像块及相关参数
    """
    # 先获得影像的长宽
    img_height = img.shape[0]
    img_width = img.shape[1]

    # 再计算按照给定的影像块大小完全覆盖全图需要多少行列
    if img_height % block_height != 0:
        row_num = int(img_height / block_height) + 1
    else:
        row_num = int(img_height / block_height)
    if img_width % block_width != 0:
        col_num = int(img_width / block_width) + 1
    else:
        col_num = int(img_width / block_width)

    # 再计算根据指定的影像块大小和行列数，全图影像应该有多大以及与原图的差异
    target_height = row_num * block_height
    target_width = col_num * block_width
    diff_height = target_height - img_height
    diff_width = target_width - img_width

    # 根据尺寸差异计算上下左右应该各自向外扩展多少
    if diff_height % 2 != 0:
        padding_top = int(diff_height / 2) + 1
    else:
        padding_top = int(diff_height / 2)
    padding_bottom = diff_height - padding_top
    if diff_width % 2 != 0:
        padding_left = int(diff_width / 2) + 1
    else:
        padding_left = int(diff_width / 2)
    padding_right = diff_width - padding_left

    # 对图像进行扩边
    img_padding = cv2.copyMakeBorder(img,
                                     padding_top, padding_bottom,
                                     padding_left, padding_right,
                                     cv2.BORDER_REFLECT)

    # 依次对每个影像块进行处理
    img_blocks = []
    block_indices = []
    counter = 0
    for i in range(row_num):
        for j in range(col_num):
            tmp_start_y = i * block_height
            tmp_start_x = j * block_width
            tmp_end_y = tmp_start_y + block_height
            tmp_end_x = tmp_start_x + block_width
            tmp_block = img_padding[tmp_start_y:tmp_end_y, tmp_start_x:tmp_end_x, :]
            img_blocks.append(tmp_block)
            block_indices.append([counter, i, j, tmp_start_x, tmp_start_y, tmp_end_x, tmp_end_y])
            counter += 1

    block_param = [img_height, img_width,
                   target_height, target_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right]

    return img_blocks, block_indices, block_param


def cropImageWithOverlapping(img, block_height, block_width, overlapping):
    """
    将输入影像划分成block_height×block_width大小的影像块并且彼此之间有overlapping的重叠(所有影像块大小相同，不足部分扩边得到) -> img_blocks, block_indices, block_param
    :param img: 输入影像
    :param block_height: 分块影像块高
    :param block_width: 分块影像块宽
    :param overlapping: 重叠部分大小
    :return: 分割好以后的影像块
    """
    # 先获得影像的长宽
    img_height = img.shape[0]
    img_width = img.shape[1]

    # 再计算按照给定的影像块大小完全覆盖全图需要多少行列
    if img_height % block_height != 0:
        row_num = int(img_height / block_height) + 1
    else:
        row_num = int(img_height / block_height)
    if img_width % block_width != 0:
        col_num = int(img_width / block_width) + 1
    else:
        col_num = int(img_width / block_width)

    # 再计算根据指定的影像块大小和行列数，全图影像应该有多大以及与原图的差异
    target_height = row_num * block_height
    target_width = col_num * block_width
    diff_height = target_height - img_height
    diff_width = target_width - img_width

    # 根据尺寸差异计算上下左右应该各自向外扩展多少
    if diff_height % 2 != 0:
        padding_top = int(diff_height / 2) + 1
    else:
        padding_top = int(diff_height / 2)
    padding_bottom = diff_height - padding_top
    if diff_width % 2 != 0:
        padding_left = int(diff_width / 2) + 1
    else:
        padding_left = int(diff_width / 2)
    padding_right = diff_width - padding_left

    # 对图像进行扩边
    img_padding = cv2.copyMakeBorder(img,
                                     padding_top, padding_bottom,
                                     padding_left, padding_right,
                                     cv2.BORDER_REFLECT)

    # 在之前的基础上进一步向外扩，满足overlapping的要求
    padding_overlap_top = padding_top + overlapping
    padding_overlap_bottom = padding_bottom + overlapping
    padding_overlap_left = padding_left + overlapping
    padding_overlap_right = padding_right + overlapping

    img_padding_overlap = cv2.copyMakeBorder(img,
                                             padding_overlap_top, padding_overlap_bottom,
                                             padding_overlap_left, padding_overlap_right,
                                             cv2.BORDER_REFLECT)
    img_padding_overlap_height = img_padding_overlap.shape[0]
    img_padding_overlap_width = img_padding_overlap.shape[1]

    # 依次对每个影像块进行处理
    img_blocks = []
    block_indices = []
    counter = 0
    for i in range(row_num):
        for j in range(col_num):
            # 这四个坐标是在处理完padding的影像上的坐标(对应img_padding)
            tmp_start_y_padding = i * block_height
            tmp_start_x_padding = j * block_width
            tmp_end_y_padding = tmp_start_y_padding + block_height
            tmp_end_x_padding = tmp_start_x_padding + block_width

            # 这四个坐标是每个影像块扩展以后在再一次扩边影像上的坐标(对应img_padding_overlap)
            tmp_start_x_overlap = tmp_start_x_padding
            tmp_start_y_overlap = tmp_start_y_padding
            tmp_end_x_overlap = tmp_end_x_padding + 2 * overlapping
            tmp_end_y_overlap = tmp_end_y_padding + 2 * overlapping

            # 不包含扩展区域的影像块
            tmp_block_padding = img_padding[tmp_start_y_padding:tmp_end_y_padding,
                                tmp_start_x_padding:tmp_end_x_padding, :]
            # 包括扩展区域的影像块
            tmp_block_overlap = img_padding_overlap[tmp_start_y_overlap:tmp_end_y_overlap,
                                tmp_start_x_overlap:tmp_end_x_overlap, :]

            img_blocks.append(tmp_block_overlap)
            block_indices.append([counter,
                                  i, j,
                                  tmp_start_x_overlap, tmp_start_y_overlap,
                                  tmp_end_x_overlap, tmp_end_y_overlap,
                                  tmp_start_x_padding, tmp_start_y_padding,
                                  tmp_end_x_padding, tmp_end_y_padding])
            counter += 1

    block_param = [img_height, img_width,
                   target_height, target_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right,
                   padding_overlap_top, padding_overlap_bottom,
                   padding_overlap_left, padding_overlap_right,
                   img_padding_overlap_height, img_padding_overlap_width,
                   overlapping]

    return img_blocks, block_indices, block_param


def saveBlocks(out_dir, out_filename, out_type, img_blocks, block_indices, block_param):
    """
    将分割好的影像块保存到文件
    :param out_dir: 输出影像块的文件夹路径
    :param out_filename: 输出影像块的文件名前缀
    :param out_type: 输出影像块文件的类型
    :param img_blocks: 待输出的影像块
    :param block_indices: 待输出影像块的索引
    :param block_param: 影像块相关参数
    :return: 无
    """
    if not out_type.__contains__("."):
        out_type = "." + out_type

    fout = open(out_dir + "/indices.txt", "w")
    fout.write("original image height:" + str(block_param[0]) + "\n")
    fout.write("original image width:" + str(block_param[1]) + "\n")
    fout.write("padding image height:" + str(block_param[2]) + "\n")
    fout.write("padding image width:" + str(block_param[3]) + "\n")
    fout.write("row num:" + str(block_param[4]) + "\n")
    fout.write("col num:" + str(block_param[5]) + "\n")
    fout.write("block height:" + str(block_param[6]) + "\n")
    fout.write("block width:" + str(block_param[7]) + "\n")
    fout.write("padding top:" + str(block_param[8]) + "\n")
    fout.write("padding bottom:" + str(block_param[9]) + "\n")
    fout.write("padding left:" + str(block_param[10]) + "\n")
    fout.write("padding right:" + str(block_param[11]) + "\n")
    fout.write("file name format:filename_rowindex_colindex.filetype\n")
    fout.write("Block indices in padding image:\n")
    fout.write("number\trow index\tcol index\tx_start\ty_start\tx_end\ty_end\n")

    for i in range(len(img_blocks)):
        index_num = block_indices[i][0]
        block_row = block_indices[i][1]
        block_col = block_indices[i][2]
        start_x = block_indices[i][3]
        start_y = block_indices[i][4]
        end_x = block_indices[i][5]
        end_y = block_indices[i][6]
        block_img = img_blocks[i]

        fout.write(index_num + "\t" +
                   str(block_row) + "\t" + str(block_col) + "\t" +
                   str(start_x) + "\t" + str(start_y) + "\t" +
                   str(end_x) + "\t" + str(end_y) + "\n")
        cv2.imwrite(out_dir + "/" + out_filename + "_" +
                    str(block_row).zfill(4) + "_" + str(block_col).zfill(4) + out_type,
                    block_img)
    fout.close()


def saveBlocksWithOverlapping(out_dir, out_filename, out_type, img_blocks, block_indices, block_param, color_mode):
    """
    将分割好的影像块保存到文件
    :param out_dir: 输出影像块的文件夹路径
    :param out_filename: 输出影像块的文件名前缀
    :param out_type: 输出影像块文件的类型
    :param img_blocks: 待输出的影像块
    :param block_indices: 待输出影像块的索引
    :param block_param: 影像块相关参数
    :param color_mode: 影像块的色彩顺序
    :return: 无
    """
    if not out_type.__contains__("."):
        out_type = "." + out_type

    fout = open(out_dir + "/indices.txt", "w")
    fout.write("original image height:" + str(block_param[0]) + "\n")
    fout.write("original image width:" + str(block_param[1]) + "\n")
    fout.write("padding image height:" + str(block_param[2]) + "\n")
    fout.write("padding image width:" + str(block_param[3]) + "\n")
    fout.write("row num:" + str(block_param[4]) + "\n")
    fout.write("col num:" + str(block_param[5]) + "\n")
    fout.write("block height:" + str(block_param[6]) + "\n")
    fout.write("block width:" + str(block_param[7]) + "\n")
    fout.write("padding top:" + str(block_param[8]) + "\n")
    fout.write("padding bottom:" + str(block_param[9]) + "\n")
    fout.write("padding left:" + str(block_param[10]) + "\n")
    fout.write("padding right:" + str(block_param[11]) + "\n")
    fout.write("padding overlap top:" + str(block_param[12]) + "\n")
    fout.write("padding overlap bottom:" + str(block_param[13]) + "\n")
    fout.write("padding overlap left:" + str(block_param[14]) + "\n")
    fout.write("padding overlap right:" + str(block_param[15]) + "\n")
    fout.write("image overlap height:" + str(block_param[16]) + "\n")
    fout.write("image overlap width:" + str(block_param[17]) + "\n")
    fout.write("overlapping:" + str(block_param[18]) + "\n")

    fout.write("file name format:filename_rowindex_colindex.filetype\n")
    fout.write("Block indices in padding image:\n")

    fout.write("number\trow index\tcol index\t"
               "x_start(overlap)\ty_start(overlap)\tx_end(overlap)\ty_end(overlap)\t"
               "x_start(padding)\ty_start(padding)\tx_end(padding)\ty_end(padding)\n")

    for i in range(len(img_blocks)):
        index_num = block_indices[i][0]
        block_row = block_indices[i][1]
        block_col = block_indices[i][2]
        start_x_overlap = block_indices[i][3]
        start_y_overlap = block_indices[i][4]
        end_x_overlap = block_indices[i][5]
        end_y_overlap = block_indices[i][6]
        start_x_padding = block_indices[i][7]
        start_y_padding = block_indices[i][8]
        end_x_padding = block_indices[i][9]
        end_y_padding = block_indices[i][10]
        block_img = img_blocks[i]

        fout.write(str(index_num) + "\t" +
                   str(block_row) + "\t" + str(block_col) + "\t" +
                   str(start_x_overlap) + "\t" + str(start_y_overlap) + "\t" +
                   str(end_x_overlap) + "\t" + str(end_y_overlap) + "\t" +
                   str(start_x_padding) + "\t" + str(start_y_padding) + "\t" +
                   str(end_x_padding) + "\t" + str(end_y_padding) + "\n")

        # 如果彩色通道顺序是RGB，就转换一下，否则不用额外操作
        if color_mode.__contains__("rgb") or color_mode.__contains__("RGB") or color_mode.__contains__("Rgb"):
            block_img = cv2.cvtColor(block_img, cv2.COLOR_RGB2BGR)
        else:
            block_img = block_img

        cv2.imwrite(out_dir + "/" + out_filename + "_" +
                    str(block_row).zfill(4) + "_" + str(block_col).zfill(4) + out_type,
                    block_img)

    fout.close()


def cropImageAndSaveBlocks(img_path, block_height, block_width, out_dir, out_filename, out_type):
    """
    cropImage()和saveBlocks()函数的二次封装
    :param img_path: 待分割的影像路径
    :param block_height: 影像块高度
    :param block_width: 影像块宽度
    :param out_dir: 影像块输出文件夹
    :param out_filename: 影像块输出前缀
    :param out_type: 影像块输出文件类型
    :return: 无
    """
    img = cv2.imread(img_path)
    img_blocks, block_indices, block_param = cropImage(img, block_height, block_width)
    saveBlocks(out_dir, out_filename, out_type, img_blocks, block_indices, block_param)


def cropImageAndSaveBlocksWithOverlapping(img_path, block_height, block_width, overlapping,
                                          out_dir, out_filename, out_type, color_mode):
    """
    cropImageWithOverlapping()和saveBlocksWithOverlapping()函数的二次封装
    :param img_path: 待分割的影像路径
    :param block_height: 影像块高度
    :param block_width: 影像块宽度
    :param overlapping: 影像块重叠区域大小
    :param out_dir: 影像块输出文件夹
    :param out_filename: 影像块输出前缀
    :param out_type: 影像块输出文件类型
    :param color_mode: 影像块的色彩顺序
    :return: 无
    """
    img = cv2.imread(img_path)
    img_blocks, block_indices, block_param = cropImageWithOverlapping(img, block_height, block_width, overlapping)
    saveBlocksWithOverlapping(out_dir, out_filename, out_type, img_blocks, block_indices, block_param, color_mode)


def loadIndexFile(file_path):
    """
    加载分块索引文件 -> block_param, block_indices
    :param file_path: 索引文件路径
    :return: 装载的影像块索引
    """
    fin = open(file_path, "r")
    original_img_height = int(fin.readline().strip().split(":")[1])
    original_img_width = int(fin.readline().strip().split(":")[1])
    padding_img_height = int(fin.readline().strip().split(":")[1])
    padding_img_width = int(fin.readline().strip().split(":")[1])
    row_num = int(fin.readline().strip().split(":")[1])
    col_num = int(fin.readline().strip().split(":")[1])
    block_height = int(fin.readline().strip().split(":")[1])
    block_width = int(fin.readline().strip().split(":")[1])
    padding_top = int(fin.readline().strip().split(":")[1])
    padding_bottom = int(fin.readline().strip().split(":")[1])
    padding_left = int(fin.readline().strip().split(":")[1])
    padding_right = int(fin.readline().strip().split(":")[1])
    padding_overlap_top = int(fin.readline().strip().split(":")[1])
    padding_overlap_bottom = int(fin.readline().strip().split(":")[1])
    padding_overlap_left = int(fin.readline().strip().split(":")[1])
    padding_overlap_right = int(fin.readline().strip().split(":")[1])
    img_padding_overlap_height = int(fin.readline().strip().split(":")[1])
    img_padding_overlap_width = int(fin.readline().strip().split(":")[1])
    overlapping = int(fin.readline().strip().split(":")[1])

    block_param = [original_img_height, original_img_width,
                   padding_img_height, padding_img_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right,
                   padding_overlap_top, padding_overlap_bottom,
                   padding_overlap_left, padding_overlap_right,
                   img_padding_overlap_height, img_padding_overlap_width,
                   overlapping]

    fin.readline()
    fin.readline()
    fin.readline()

    block_indices = []
    line = fin.readline().strip()
    while line:
        line_parts = line.split("\t")
        index_num = int(line_parts[0])
        row_index = int(line_parts[1])
        col_index = int(line_parts[2])
        x_start_overlap = int(line_parts[3])
        y_start_overlap = int(line_parts[4])
        x_end_overlap = int(line_parts[5])
        y_end_overlap = int(line_parts[6])
        x_start_padding = int(line_parts[7])
        y_start_padding = int(line_parts[8])
        x_end_padding = int(line_parts[9])
        y_end_padding = int(line_parts[10])
        block_indices.append([index_num,
                              row_index, col_index,
                              x_start_overlap, y_start_overlap,
                              x_end_overlap, y_end_overlap,
                              x_start_padding, y_start_padding,
                              x_end_padding, y_end_padding])

        line = fin.readline().strip()
    fin.close()
    return block_param, block_indices


def loadBlocks(block_dir, block_type):
    """
    加载分块好的影像块 -> img_blocks, block_indices, block_param
    :param block_dir: 影像块所在文件夹路径
    :param block_type: 影像块文件类型
    :return: 解析好的影像块内容及索引
    """
    fin = open(block_dir + "/indices.txt", "r")
    original_img_height = int(fin.readline().strip().split(":")[1])
    original_img_width = int(fin.readline().strip().split(":")[1])
    padding_img_height = int(fin.readline().strip().split(":")[1])
    padding_img_width = int(fin.readline().strip().split(":")[1])
    row_num = int(fin.readline().strip().split(":")[1])
    col_num = int(fin.readline().strip().split(":")[1])
    block_height = int(fin.readline().strip().split(":")[1])
    block_width = int(fin.readline().strip().split(":")[1])
    padding_top = int(fin.readline().strip().split(":")[1])
    padding_bottom = int(fin.readline().strip().split(":")[1])
    padding_left = int(fin.readline().strip().split(":")[1])
    padding_right = int(fin.readline().strip().split(":")[1])

    block_param = [original_img_height, original_img_width,
                   padding_img_height, padding_img_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right]

    fin.readline()
    fin.readline()
    fin.readline()

    block_indices = []
    line = fin.readline().strip()
    while line:
        line_parts = line.split("\t")
        index_num = int(line_parts[0])
        row_index = int(line_parts[1])
        col_index = int(line_parts[2])
        x_start = int(line_parts[3])
        y_start = int(line_parts[4])
        x_end = int(line_parts[5])
        y_end = int(line_parts[6])
        block_indices.append([index_num, row_index, col_index, x_start, y_start, x_end, y_end])

        line = fin.readline().strip()

    img_blocks = []
    paths, names, files = common.findFiles(block_dir, block_type)
    for i in range(len(files)):
        img_block = cv2.imread(files[i])
        img_blocks.append(img_block)

    fin.close()

    return img_blocks, block_indices, block_param


def loadBlocksWithOverlapping(block_dir, block_type):
    """
    加载分块好的影像块 -> img_blocks, block_indices, block_param
    :param block_dir: 影像块所在文件夹路径
    :param block_type: 影像块文件类型
    :return: 解析好的影像块内容及索引
    """
    fin = open(block_dir + "/indices.txt", "r")
    original_img_height = int(fin.readline().strip().split(":")[1])
    original_img_width = int(fin.readline().strip().split(":")[1])
    padding_img_height = int(fin.readline().strip().split(":")[1])
    padding_img_width = int(fin.readline().strip().split(":")[1])
    row_num = int(fin.readline().strip().split(":")[1])
    col_num = int(fin.readline().strip().split(":")[1])
    block_height = int(fin.readline().strip().split(":")[1])
    block_width = int(fin.readline().strip().split(":")[1])
    padding_top = int(fin.readline().strip().split(":")[1])
    padding_bottom = int(fin.readline().strip().split(":")[1])
    padding_left = int(fin.readline().strip().split(":")[1])
    padding_right = int(fin.readline().strip().split(":")[1])
    padding_overlap_top = int(fin.readline().strip().split(":")[1])
    padding_overlap_bottom = int(fin.readline().strip().split(":")[1])
    padding_overlap_left = int(fin.readline().strip().split(":")[1])
    padding_overlap_right = int(fin.readline().strip().split(":")[1])
    img_padding_overlap_height = int(fin.readline().strip().split(":")[1])
    img_padding_overlap_width = int(fin.readline().strip().split(":")[1])
    overlapping = int(fin.readline().strip().split(":")[1])

    block_param = [original_img_height, original_img_width,
                   padding_img_height, padding_img_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right,
                   padding_overlap_top, padding_overlap_bottom,
                   padding_overlap_left, padding_overlap_right,
                   img_padding_overlap_height, img_padding_overlap_width,
                   overlapping]

    fin.readline()
    fin.readline()
    fin.readline()

    block_indices = []
    line = fin.readline().strip()
    while line:
        line_parts = line.split("\t")
        index_num = int(line_parts[0])
        row_index = int(line_parts[1])
        col_index = int(line_parts[2])
        x_start_overlap = int(line_parts[3])
        y_start_overlap = int(line_parts[4])
        x_end_overlap = int(line_parts[5])
        y_end_overlap = int(line_parts[6])
        x_start_padding = int(line_parts[7])
        y_start_padding = int(line_parts[8])
        x_end_padding = int(line_parts[9])
        y_end_padding = int(line_parts[10])
        block_indices.append([index_num,
                              row_index, col_index,
                              x_start_overlap, y_start_overlap,
                              x_end_overlap, y_end_overlap,
                              x_start_padding, y_start_padding,
                              x_end_padding, y_end_padding])

        line = fin.readline().strip()

    img_blocks = []
    paths, names, files = common.findFiles(block_dir, block_type)
    for i in range(len(files)):
        img_block = cv2.imread(files[i])
        img_blocks.append(img_block)

    fin.close()

    return img_blocks, block_indices, block_param


def mergeBlocks(img_blocks, block_indices, block_param):
    """
    将影像块合并成一张大图 -> padding_img, crop_img
    :param img_blocks: 待合并的影像块
    :param block_indices: 对应的影像块索引
    :param block_param: 影像块相关参数
    :return: padding_img, crop_img, 其中crop_img是和原图大小相同的影像
    """
    padding_img_height = block_param[2]
    padding_img_width = block_param[3]
    padding_img = np.zeros([padding_img_height, padding_img_width, 3], np.uint8)

    for i in range(len(img_blocks)):
        tmp_start_x = block_indices[i][2]
        tmp_start_y = block_indices[i][3]
        tmp_end_x = block_indices[i][4]
        tmp_end_y = block_indices[i][5]
        padding_img[tmp_start_y:tmp_end_y, tmp_start_x:tmp_end_x, :] = img_blocks[i]
    padding_top = block_param[8]
    padding_bottom = block_param[9]
    padding_left = block_param[10]
    padding_right = block_param[11]

    crop_img = padding_img[padding_top:padding_img_height - padding_bottom,
               padding_left:padding_img_width - padding_right, :]
    return padding_img, crop_img


def mergeBlocksWithOverlapping(img_blocks, block_indices, block_param):
    """
    将影像块合并成一张大图 -> overlapping_img, padding_img, original_img
    :param img_blocks: 待合并的影像块
    :param block_indices: 对应的影像块索引
    :param block_param: 影像块相关参数
    :return: overlapping_img, padding_img, original_img, 其中original_img是和原图大小相同的影像
    """
    overlapping_img_height = block_param[16]
    overlapping_img_width = block_param[17]
    overlapping_img = np.zeros([overlapping_img_height, overlapping_img_width, 3], np.uint8)

    padding_img_height = block_param[2]
    padding_img_width = block_param[3]
    padding_img = np.zeros([padding_img_height, padding_img_width, 3], np.uint8)

    padding_top = block_param[8]
    padding_bottom = block_param[9]
    padding_left = block_param[10]
    padding_right = block_param[11]

    overlapping = block_param[18]

    for i in range(len(img_blocks)):
        start_x_overlap = block_indices[i][3]
        start_y_overlap = block_indices[i][4]
        end_x_overlap = block_indices[i][5]
        end_y_overlap = block_indices[i][6]

        start_x_padding = block_indices[i][7]
        start_y_padding = block_indices[i][8]
        end_x_padding = block_indices[i][9]
        end_y_padding = block_indices[i][10]

        block_overlap = img_blocks[i]
        block_padding = img_blocks[i][overlapping:block_overlap.shape[0] - overlapping,
                        overlapping:block_overlap.shape[1] - overlapping, :]

        # 对于重叠影像，直接贴过来（会有拼接缝，如果有更好的融合方法可以尝试）
        overlapping_img[start_y_overlap:end_y_overlap, start_x_overlap:end_x_overlap, :] = block_overlap
        # 对于非重叠影像，裁剪影像块之后再贴过来（无拼接缝）
        padding_img[start_y_padding:end_y_padding, start_x_padding:end_x_padding, :] = block_padding

    # 对于原始影像，直接在padding影像上裁剪即可
    original_img = padding_img[padding_top:padding_img.shape[0] - padding_bottom,
                   padding_left:padding_img.shape[1] - padding_right, :]
    return overlapping_img, padding_img, original_img


def mergeBlocksAndSaveImage(block_dir, block_type, out_dir, out_type):
    """
    loadBlocks()函数和mergeBlocks()函数的二次封装
    :param block_dir: 影像块所在路径
    :param block_type: 影像块文件类型
    :param out_dir: 合并影像的输出文件夹路径
    :param out_type: 合并影像的类型
    :return: 无
    """
    if not out_type.__contains__("."):
        out_type = "." + out_type
    img_blocks, block_indices, block_param = loadBlocks(block_dir, block_type)
    padding_img, crop_img = mergeBlocks(img_blocks, block_indices, block_param)
    cv2.imwrite(out_dir + "/" + "merge_padding" + out_type, padding_img)
    cv2.imwrite(out_dir + "/" + "merge_original" + out_type, crop_img)


def mergeBlocksAndSaveImageWithOverlapping(block_dir, block_type, out_dir, out_type):
    """
    loadBlocks()函数和mergeBlocks()函数的二次封装
    :param block_dir: 影像块所在路径
    :param block_type: 影像块文件类型
    :param out_dir: 合并影像的输出文件夹路径
    :param out_type: 合并影像的类型
    :return: 无
    """
    if not out_type.__contains__("."):
        out_type = "." + out_type
    img_blocks, block_indices, block_param = loadBlocksWithOverlapping(block_dir, block_type)
    overlapping_img, crop_img, original_img = mergeBlocksWithOverlapping(img_blocks, block_indices, block_param)
    cv2.imwrite(out_dir + "/" + "merge_overlapping" + out_type, overlapping_img)
    cv2.imwrite(out_dir + "/" + "merge_cropping" + out_type, crop_img)
    cv2.imwrite(out_dir + "/" + "merge_original" + out_type, original_img)


def reverseRGB(img):
    """
    反转RGB波段顺序 -> img2
    :param img: RGB波段影像
    :return: 波段顺序反转的波段影像
    """

    img2 = np.zeros(img.shape, img.dtype)
    img2[:, :, 0] = img[:, :, 2]
    img2[:, :, 1] = img[:, :, 1]
    img2[:, :, 2] = img[:, :, 0]
    return img2


def splitRGB(img):
    """
    用于对读取的RGB影像进行波段拆分，返回单独的每个波段 -> band_r, band_g, band_b
    :param img: RGB影像
    :return: 拆分后的R、G、B波段数据
    """

    band_r = img[:, :, 0]
    band_g = img[:, :, 1]
    band_b = img[:, :, 2]
    return band_r, band_g, band_b


def mergeRGB(band_r, band_g, band_b):
    """
    合并独立的R、G、B波段数据为一个彩色图像 -> img
    :param band_r: R波段
    :param band_g: G波段
    :param band_b: B波段
    :return: 彩色图像
    """

    h = min(band_r.shape[0], band_g.shape[0], band_b.shape[0])
    w = min(band_r.shape[1], band_g.shape[1], band_b.shape[1])
    img = np.zeros([h, w, 3], np.uint8)
    img[:, :, 0] = band_r[:h, :w]
    img[:, :, 1] = band_g[:h, :w]
    img[:, :, 2] = band_b[:h, :w]
    return img


def getSurfKps(img, hessianTh=1500):
    """
    获取SURF特征点和描述子 -> kp, des
    :param img: 读取的输入影像
    :param hessianTh: 海塞矩阵阈值，默认为1500
    :return: 特征点和对应的描述子
    """

    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=hessianTh)
    kp, des = cv2.xfeatures2d_SURF.detectAndCompute(surf, img, None)
    return kp, des


def getSiftKps(img, numKps=2000):
    """
    获取SIFT特征点和描述子 -> kp, des
    :param img: 读取的输入影像
    :param numKps: 期望提取的特征点个数，默认2000
    :return: 特征点和对应的描述子
    """

    sift = cv2.xfeatures2d_SIFT.create(nfeatures=numKps)
    kp, des = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img, None)
    return kp, des


def getOrbKps(img, numKps=2000):
    """
    获取ORB特征点和描述子 -> kp, des
    :param img: 读取的输入影像
    :param numKps: 期望提取的特征点个数，默认2000
    :return: 特征点和对应的描述子
    """

    orb = cv2.ORB_create(nfeatures=numKps)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def getMixKps(img, numKps=1000):
    """
    获取mix特征点 -> kps_final, des_final
    :param img: 输入影像
    :param numKps: 期望特征点个数
    :return: 特征点与描述子
    """
    sift = cv2.xfeatures2d_SIFT.create(nfeatures=numKps)
    kps1 = cv2.xfeatures2d_SIFT.detect(sift, img, None)
    bf = cv2.xfeatures2d_BriefDescriptorExtractor.create()
    kps_final, des_final = cv2.xfeatures2d_BriefDescriptorExtractor.compute(bf, img, kps1, None)
    return kps_final, des_final


def getMixKpsPrivate(sift, bf, img, numKps=1000):
    kps1 = cv2.xfeatures2d_SIFT.detect(sift, img, None)
    kps_final, des_final = cv2.xfeatures2d_BriefDescriptorExtractor.compute(bf, img, kps1, None)
    return kps_final, des_final


def siftSpeedUp(input_data):
    """
    基于多进程并行的Sift特征提取加速函数，供内部函数调用
    :param input_data: 一个元组，包含(影像块,块范围索引,每块特征点个数)
    :return: 返回一个元组，包含特征点和描述子
    """

    img_block = input_data[0]
    block_range = input_data[1]
    num = input_data[2]
    kp, des = getSiftKps(img_block, numKps=num)
    kps = []
    if type(kp[0]) is cv2.KeyPoint:
        tmp_kps = cvtCvKeypointToNormal(kp)
        for i in range(tmp_kps.__len__()):
            kps.append((tmp_kps[i][0] + block_range[2], tmp_kps[i][1] + block_range[0]))
    else:
        for i in range(kp.__len__()):
            kps.append((kp[i][0] + block_range[2], kp[i][1] + block_range[0]))
    return (kps, des)


def mixSpeedUp(input_data):
    """
    基于多进程并行的mix特征提取加速函数，供内部函数调用
    :param input_data: 一个元组，包含(影像块,块范围索引,每块特征点个数)
    :return: 返回一个元组，包含特征点和描述子
    """

    img_block = input_data[0]
    block_range = input_data[1]
    num = input_data[2]
    kp, des = getMixKps(img_block, numKps=num)
    kps = []
    if type(kp[0]) is cv2.KeyPoint:
        tmp_kps = cvtCvKeypointToNormal(kp)
        for i in range(tmp_kps.__len__()):
            kps.append((tmp_kps[i][0] + block_range[2], tmp_kps[i][1] + block_range[0]))
    else:
        for i in range(kp.__len__()):
            kps.append((kp[i][0] + block_range[2], kp[i][1] + block_range[0]))
    return (kps, des)


def surfSpeedUp(input_data):
    """
    基于多进程并行的Surf特征提取加速函数，供内部函数调用
    :param input_data: 一个元组，包含(影像块,块范围索引,海塞矩阵阈值)
    :return: 返回一个元组，包含特征点和描述子
    """

    img_block = input_data[0]
    block_range = input_data[1]
    thHessian = input_data[2]
    kp, des = getSurfKps(img_block, hessianTh=thHessian)
    kps = []
    if type(kp[0]) is cv2.KeyPoint:
        tmp_kps = cvtCvKeypointToNormal(kp)
        for i in range(tmp_kps.__len__()):
            kps.append((tmp_kps[i][0] + block_range[2], tmp_kps[i][1] + block_range[0]))
    else:
        for i in range(kp.__len__()):
            kps.append((kp[i][0] + block_range[2], kp[i][1] + block_range[0]))
    return (kps, des)


def orbSpeedUp(input_data):
    """
    基于多进程并行的Orb特征提取加速函数，供内部函数调用
    :param input_data: 一个元组，包含(影像块,块范围索引,每块特征点个数)
    :return: 返回一个元组，包含特征点和描述子
    """

    img_block = input_data[0]
    block_range = input_data[1]
    num = input_data[2]
    kp, des = getOrbKps(img_block, numKps=num)
    kps = []
    if type(kp[0]) is cv2.KeyPoint:
        tmp_kps = cvtCvKeypointToNormal(kp)
        for i in range(tmp_kps.__len__()):
            kps.append((tmp_kps[i][0] + block_range[2], tmp_kps[i][1] + block_range[0]))
    else:
        for i in range(kp.__len__()):
            kps.append((kp[i][0] + block_range[2], kp[i][1] + block_range[0]))
    return (kps, des)


def getSiftKpsWithBlockSpeedUp(img, row=2, col=2, kpsPerBlock=2000, processNum=4):
    """
    多进程并行的分块Sift提取加速函数
    :param img: 原始完整影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块提取的特征点数量，默认为2000
    :param processNum: 并行进程数，默认为4，且最大不大于全部分块数
    :return: 提取的全图特征点和描述子
    """

    parts = getBlockRange(img, row=row, col=col)
    input_data = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        input_data.append((img_part, parts[i], kpsPerBlock))
    # if processNum > row * col:
    #     pool = Pool(processes=row * col)
    # else:
    #     pool = Pool(processes=processNum)
    pool = Pool(processes=processNum)
    res = pool.map(siftSpeedUp, input_data)
    pool.close()
    pool.join()

    kps_list = []
    des_list = []
    for i in range(res.__len__()):
        kps_list.extend(res[i][0])
        des_list.append(res[i][1])
    des = np.vstack(tuple(des_list))
    return kps_list, des


def getMixKpsWithBlockSpeedUp(img, row=2, col=2, kpsPerBlock=2000, processNum=4):
    """
    多进程并行的分块Mix提取加速函数
    :param img: 原始完整影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块提取的特征点数量，默认为2000
    :param processNum: 并行进程数，默认为4，且最大不大于全部分块数
    :return: 提取的全图特征点和描述子
    """

    parts = getBlockRange(img, row=row, col=col)
    input_data = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        input_data.append((img_part, parts[i], kpsPerBlock))
    # if processNum > row * col:
    #     pool = Pool(processes=row * col)
    # else:
    #     pool = Pool(processes=processNum)
    pool = Pool(processes=processNum)
    res = pool.map(mixSpeedUp, input_data)
    pool.close()
    pool.join()

    kps_list = []
    des_list = []
    for i in range(res.__len__()):
        kps_list.extend(res[i][0])
        des_list.append(res[i][1])
    des = np.vstack(tuple(des_list))
    return kps_list, des


def getSurfKpsWithBlockSpeedUp(img, row=2, col=2, thHessian=1500, processNum=4):
    """
    多进程并行的分块Surf提取加速函数
    :param img: 原始完整影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param thHessian: Surf算子的海塞矩阵阈值，默认为1500
    :param processNum: 并行进程数，默认为4，且最大不大于全部分块数
    :return: 提取的全图特征点和描述子
    """

    parts = getBlockRange(img, row=row, col=col)
    input_data = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        input_data.append((img_part, parts[i], thHessian))
    if processNum > row * col:
        pool = Pool(processes=row * col)
    else:
        pool = Pool(processes=processNum)
    res = pool.map(surfSpeedUp, input_data)
    pool.close()
    pool.join()

    kps_list = []
    des_list = []
    for i in range(res.__len__()):
        kps_list.extend(res[i][0])
        des_list.append(res[i][1])
    des = np.vstack(tuple(des_list))
    return kps_list, des


def getOrbKpsWithBlockSpeedUp(img, row=2, col=2, kpsPerBlock=2000, processNum=4):
    """
    多进程并行的分块Orb提取加速函数
    :param img: 原始完整影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块的特征点个数，默认为2000
    :param processNum: 并行进程数，默认为4，且最大不大于全部分块数
    :return: 提取的全图特征点和描述子
    """

    parts = getBlockRange(img, row=row, col=col)
    input_data = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        input_data.append((img_part, parts[i], kpsPerBlock))
    if processNum > row * col:
        pool = Pool(processes=row * col)
    else:
        pool = Pool(processes=processNum)
    res = pool.map(orbSpeedUp, input_data)
    pool.close()
    pool.join()

    kps_list = []
    des_list = []
    for i in range(res.__len__()):
        kps_list.extend(res[i][0])
        des_list.append(res[i][1])
    des = np.vstack(tuple(des_list))
    return kps_list, des


def getSiftKpsWithBlock(img, row=2, col=2, kpsPerBlock=2000):
    """
    分块Sift特征提取函数，对于较大影像比较有效果
    :param img: 原始影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块的特征点个数，默认为2000
    :return: 提取的全图特征点和描述子
    """

    parts = getBlockRange(img, row=row, col=col)
    kp = []
    de = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        tmp_kp, tmp_des = getSiftKps(img_part, numKps=kpsPerBlock)
        kp.append(tmp_kp)
        de.append(tmp_des)
    kps, des = joinKps(kp, de, parts)
    return kps, des


def getMixKpsWithBlock(img, row=2, col=2, kpsPerBlock=2000):
    """
    分块Mix特征提取函数，对于较大影像比较有效果
    :param img: 原始影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块的特征点个数，默认为2000
    :return: 提取的全图特征点和描述子
    """

    parts = getBlockRange(img, row=row, col=col)
    kp = []
    de = []
    sift = cv2.xfeatures2d_SIFT.create(nfeatures=kpsPerBlock)
    bf = cv2.xfeatures2d_BriefDescriptorExtractor.create()
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        tmp_kp, tmp_des = getMixKpsPrivate(sift, bf, img_part, numKps=kpsPerBlock)
        kp.append(tmp_kp)
        de.append(tmp_des)
    kps, des = joinKps(kp, de, parts)
    return kps, des


def getSurfKpsWithBlock(img, row=2, col=2, thHessian=1500):
    """
    分块Surf特征提取函数，对于较大影像比较有效果
    :param img: 原始影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param thHessian: Surf的海塞矩阵阈值，默认为1500
    :return: 提取的全图特征点和描述子
    """

    parts = getBlockRange(img, row=row, col=col)
    kp = []
    de = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        tmp_kp, tmp_des = getSurfKps(img_part, hessianTh=thHessian)
        kp.append(tmp_kp)
        de.append(tmp_des)
    kps, des = joinKps(kp, de, parts)
    return kps, des


def getORBKpsWithBlock(img, row=2, col=2, kpsPerBlock=2000):
    """
    分块Orb特征提取函数，对于较大影像比较有效果
    :param img: 原始影像
    :param row: 分块的行数，默认为2
    :param col: 分块的列数，默认为2
    :param kpsPerBlock: 每块的特征点数量，默认为2000
    :return: 提取的全图特征点和描述子
    """

    parts = getBlockRange(img, row=row, col=col)
    kp = []
    de = []
    for i in range(parts.__len__()):
        img_part = img[parts[i][0]:parts[i][1], parts[i][2]:parts[i][3]]
        tmp_kp, tmp_des = getOrbKps(img_part, numKps=kpsPerBlock)
        kp.append(tmp_kp)
        de.append(tmp_des)
    kps, des = joinKps(kp, de, parts)
    return kps, des


def joinKps(kps_list, des_list, parts):
    """
    对于分块模式，用于将每一块提取出的特征点列表和描述子合并成一个list和描述子
    :param kps_list: 包含有多个特征点列表的列表
    :param des_list: 包含有多个描述子的列表
    :param parts: 分块索引列表，通过getBlockRange函数获得
    :return: 合并后的特征点列表和对应描述子
    """

    kps = []

    for i in range(kps_list.__len__()):
        if type(kps_list[i][0]) is cv2.KeyPoint:
            tmp_kps = cvtCvKeypointToNormal(kps_list[i])
            print(parts[i][0], parts[i][2])
            for j in range(tmp_kps.__len__()):
                kps.append((tmp_kps[j][0] + parts[i][2], tmp_kps[j][1] + parts[i][0]))
        else:
            for j in range(kps_list[i].__len__()):
                kps.append((kps_list[i][j][0] + parts[i][2], kps_list[i][j][1] + parts[i][0]))
    des = np.vstack(tuple(des_list))
    return kps, des


def cvtCvKeypointToNormal(keypoints):
    """
    将OpenCV中KeyPoint类型的特征点转换成(x,y)格式的普通数值 -> cvt_kps
    :param keypoints: KeyPoint类型的特征点列表
    :return: 转换后的普通特征点列表
    """

    cvt_kps = []
    for i in range(keypoints.__len__()):
        cvt_kps.append((keypoints[i].pt[0], keypoints[i].pt[1]))
    return cvt_kps


def saveKeyPoints(keypoints, savePath, seprator='\t'):
    """
    将特征点输出到文本文件中
    :param keypoints: 特征点列表
    :param savePath: 输出文件路径和文件名
    :param seprator: 可选参数，数据分隔符，默认为tab
    :return: 空
    """

    if keypoints.__len__() != 0:
        save_file = open(savePath, 'w+')
        save_file.write(keypoints.__len__().__str__() + "\n")
        if type(keypoints[0]) is cv2.KeyPoint:
            kps = cvtCvKeypointToNormal(keypoints)
            for i in range(kps.__len__()):
                save_file.write(kps[i][0].__str__() + seprator + kps[i][1].__str__() + "\n")
        else:
            for i in range(keypoints.__len__()):
                save_file.write(keypoints[i][0].__str__() + seprator + keypoints[i][1].__str__() + "\n")
        save_file.close()
    else:
        print("keypoint list is empty,please check.")


def readKeypoints(filePath, separator='\t'):
    """
    读取保存到文件的特征点坐标数据(x,y) -> kps
    :param filePath: 数据文件路径
    :param separator: 数据分隔符，默认为一个tab
    :return: 读取到的特征点数据list
    """
    kps = []
    open_file = open(filePath, 'r')
    num = open_file.readline().strip()
    data_line = open_file.readline().strip()
    while data_line:
        split_parts = data_line.split(separator)
        kps.append((float(split_parts[0]), float(split_parts[1])))
        data_line = open_file.readline().strip()
    return kps


def drawKeypoints(img, kps, color=[0, 0, 255], rad=3, thickness=1):
    """
    在影像上绘制特征点
    :param img: 待绘制的影像
    :param kps: 待绘制的特征点列表
    :param color: 特征点颜色，默认红色，-1表示随机
    :param rad: 特征点大小，默认为3
    :param thickness: 特征点轮廓粗细，默认为1
    :return: 带有特征点的影像
    """

    if img.shape.__len__() == 2:
        img_pro = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_pro = img.copy()
    if color[0] == -1 and color[1] == -1 and color[2] == -1:
        if type(kps[0]) is cv2.KeyPoint:
            for point in kps:
                pt = (int(point.pt[0]), int(point.pt[1]))
                color[0] = common.getRandomNum(0, 255)
                color[1] = common.getRandomNum(0, 255)
                color[2] = common.getRandomNum(0, 255)
                cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
        else:
            for point in kps:
                pt = (int(point[0]), int(point[1]))
                color[0] = common.getRandomNum(0, 255)
                color[1] = common.getRandomNum(0, 255)
                color[2] = common.getRandomNum(0, 255)
                cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
    else:
        if type(kps[0]) is cv2.KeyPoint:
            for point in kps:
                pt = (int(point.pt[0]), int(point.pt[1]))
                cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
        else:
            for point in kps:
                pt = (int(point[0]), int(point[1]))
                cv2.circle(img_pro, pt, rad, color, thickness, cv2.LINE_AA)
    return img_pro


def drawAndSaveKeypoints(img, kps, save_path, color=[0, 0, 255], rad=3, thickness=1):
    """
    drawKeypoints()的二次封装，直接保存图片
    :param img: 待绘制的影像
    :param kps: 待绘制的特征点列表
    :param save_path: 输出影像的路径
    :param color: 特征点颜色，默认红色，-1表示随机
    :param rad: 特征点大小，默认为3
    :param thickness: 特征点轮廓粗细，默认为1
    :return: 无
    """
    kp_img = drawKeypoints(img, kps, color, rad, thickness)
    cv2.imwrite(save_path, kp_img)


def logTransform(img, v=200, c=256):
    """
    影像的灰度对数拉伸变换，默认支持8bit灰度 -> img_new
    :param img: 待变换影像
    :param v: 变换系数v，越大效果越明显
    :param c: 灰度量化级数的最大值，默认为8bit(256)
    :return: 拉伸后的影像
    """

    img_normalize = img * 1.0 / c
    log_res = c * (np.log(1 + v * img_normalize) / np.log(v + 1))
    img_new = np.uint8(log_res)
    return img_new


def flannMatch(kp1, des1, kp2, des2):
    """
    基于FLANN算法的匹配 -> good_kps1, good_kps2
    :param kp1: 特征点列表1
    :param des1: 特征点描述列表1
    :param kp2: 特征点列表2
    :param des2: 特征点描述列表2
    :return: 匹配的特征点对
    """

    good_matches = []
    good_kps1 = []
    good_kps2 = []

    print("kp1 num:" + len(kp1).__str__() + "," + "kp2 num:" + len(kp2).__str__())

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    cvt_kp1 = []
    cvt_kp2 = []
    if type(kp1[0]) is cv2.KeyPoint:
        cvt_kp1 = cvtCvKeypointToNormal(kp1)
    else:
        cvt_kp1 = kp1

    if type(kp2[0]) is cv2.KeyPoint:
        cvt_kp2 = cvtCvKeypointToNormal(kp2)
    else:
        cvt_kp2 = kp2

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append([cvt_kp1[matches[i][0].queryIdx][0], cvt_kp1[matches[i][0].queryIdx][1]])
            good_kps2.append([cvt_kp2[matches[i][0].trainIdx][0], cvt_kp2[matches[i][0].trainIdx][1]])

    if good_matches.__len__() == 0:
        print("No enough good matches.")
        return good_kps1, good_kps2
    else:
        print("good matches:" + good_matches.__len__().__str__())
        return good_kps1, good_kps2


def bfMatch(kp1, des1, kp2, des2, disTh=15.0):
    """
    基于BF算法的匹配 -> good_kps1, good_kps2
    :param kp1: 特征点列表1
    :param des1: 特征点描述列表1
    :param kp2: 特征点列表2
    :param des2: 特征点描述列表2
    :return: 匹配的特征点对
    """

    good_kps1 = []
    good_kps2 = []
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    if matches.__len__() == 0:
        return good_kps1, good_kps2
    else:
        min_dis = 10000
        for item in matches:
            dis = item.distance
            if dis < min_dis:
                min_dis = dis

        g_matches = []
        for match in matches:
            if match.distance <= max(1.1 * min_dis, disTh):
                g_matches.append(match)

        print("matches:" + g_matches.__len__().__str__())

        cvt_kp1 = []
        cvt_kp2 = []
        if type(kp1[0]) is cv2.KeyPoint:
            cvt_kp1 = cvtCvKeypointToNormal(kp1)
        else:
            cvt_kp1 = kp1
        if type(kp2[0]) is cv2.KeyPoint:
            cvt_kp2 = cvtCvKeypointToNormal(kp2)
        else:
            cvt_kp2 = kp2

        for i in range(g_matches.__len__()):
            good_kps1.append([cvt_kp1[g_matches[i].queryIdx][0], cvt_kp1[g_matches[i].queryIdx][1]])
            good_kps2.append([cvt_kp2[g_matches[i].trainIdx][0], cvt_kp2[g_matches[i].trainIdx][1]])

        return good_kps1, good_kps2


def siftFlannMatch(img1, img2, numKps=2000):
    """
    包装的函数，直接用于sift匹配，方便使用 -> good_kp1, good_kp2
    :param img1: 输入影像1
    :param img2: 输入影像2
    :param numKps: 每张影像上期望提取的特征点数量，默认为2000
    :return: 匹配好的特征点列表
    """

    kp1, des1 = getSiftKps(img1, numKps=numKps)
    kp2, des2 = getSiftKps(img2, numKps=numKps)
    good_kp1, good_kp2 = flannMatch(kp1, des1, kp2, des2)
    return good_kp1, good_kp2


def surfFlannMatch(img1, img2, thHessian=1500):
    """
    包装的函数，直接用于surf匹配，方便使用 -> good_kp1, good_kp2
    :param img1: 输入影像1
    :param img2: 输入影像2
    :param thHessian: 海塞矩阵阈值，默认为1500
    :return: 匹配好的特征点列表
    """

    kp1, des1 = getSurfKps(img1, hessianTh=thHessian)
    kp2, des2 = getSurfKps(img2, hessianTh=thHessian)
    good_kp1, good_kp2 = flannMatch(kp1, des1, kp2, des2)
    return good_kp1, good_kp2


def orbBFMatch(img1, img2, numKps=2000, disTh=15.0):
    """
    包装的函数，直接用于orb匹配，方便使用 -> good_kp1, good_kp2
    :param img1: 输入影像1
    :param img2: 输入影像2
    :param numKps: 每张影像上期望提取的特征点数量，默认为2000
    :return: 匹配好的特征点列表
    """

    kp1, des1 = getOrbKps(img1, numKps=numKps)
    kp2, des2 = getOrbKps(img2, numKps=numKps)
    good_kp1, good_kp2 = bfMatch(kp1, des1, kp2, des2, disTh=disTh)
    return good_kp1, good_kp2


def drawMatches(img1, kps1, img2, kps2, color=[0, 0, 255], rad=5, thickness=1):
    """
    用于绘制两幅影像间的匹配点对 -> img_out
    :param img1: 影像1
    :param kps1: 影像1上匹配的特征点
    :param img2: 影像2
    :param kps2: 影像2上匹配的特征点
    :param color: 特征点及连线的颜色，默认为红色，-1表示随机
    :param rad: 特征点大小，默认为5
    :param thickness: 匹配连线的宽度，默认为1
    :return: 绘制好的特征点匹配影像
    """

    if img1.shape.__len__() == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.shape.__len__() == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
    img_out[:img1.shape[0], :img1.shape[1], :] = img1
    img_out[:img2.shape[0], img1.shape[1]:, :] = img2

    cvt_kps1 = []
    cvt_kps2 = []
    if type(kps1[0]) == cv2.KeyPoint:
        for kp1 in kps1:
            cvt_kps1.append((int(kp1.pt[0]), int(kp1.pt[1])))
    else:
        for kp1 in kps1:
            cvt_kps1.append((int(kp1[0]), int(kp1[1])))
    if type(kps2[0]) == cv2.KeyPoint:
        for kp2 in kps2:
            cvt_kps2.append((int(kp2.pt[0] + img1.shape[1]), int(kp2.pt[1])))
    else:
        for kp2 in kps2:
            cvt_kps2.append((int(kp2[0] + img1.shape[1]), int(kp2[1])))

    if color[0] == -1 and color[1] == -1 and color[2] == -1:
        for pt1, pt2 in zip(cvt_kps1, cvt_kps2):
            color[0] = common.getRandomNum(0, 255)
            color[1] = common.getRandomNum(0, 255)
            color[2] = common.getRandomNum(0, 255)
            cv2.circle(img_out, pt1, rad, color, thickness, cv2.LINE_AA)
            cv2.circle(img_out, pt2, rad, color, thickness, cv2.LINE_AA)
            cv2.line(img_out, pt1, pt2, color, thickness, cv2.LINE_AA)
    else:
        for pt1, pt2 in zip(cvt_kps1, cvt_kps2):
            cv2.circle(img_out, pt1, rad, color, thickness, cv2.LINE_AA)
            cv2.circle(img_out, pt2, rad, color, thickness, cv2.LINE_AA)
            cv2.line(img_out, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img_out


def saveMatchPoints(keypoints1, keypoints2, savePath, seprator='\t'):
    """
    将匹配的特征点输出到文本文件中
    :param keypoints1: 特征点列表1
    :param keypoints2: 特征点列表2
    :param savePath: 输出文件路径和文件名
    :param seprator: 可选参数，数据分隔符，默认为tab
    :return: 空
    """

    if keypoints1.__len__() != 0 and keypoints2.__len__() != 0:
        save_file = open(savePath, 'w+')
        save_file.write(keypoints1.__len__().__str__() + "\n")
        if type(keypoints1[0]) is cv2.KeyPoint:
            kps1 = cvtCvKeypointToNormal(keypoints1)
            kps2 = cvtCvKeypointToNormal(keypoints2)
            for i in range(kps1.__len__()):
                save_file.write(kps1[i][0].__str__() + seprator + kps1[i][1].__str__() + seprator +
                                kps2[i][0].__str__() + seprator + kps2[i][1].__str__() + "\n")
        else:
            for i in range(keypoints1.__len__()):
                save_file.write(keypoints1[i][0].__str__() + seprator + keypoints1[i][1].__str__() + seprator +
                                keypoints2[i][0].__str__() + seprator + keypoints2[i][1].__str__() + "\n")
        save_file.close()
    else:
        print("keypoint list is empty,please check.")


def readMatchPoints(filePath, separator='\t'):
    """
    读取保存的匹配点信息，格式为x1 y1 x2 y2 -> kps1, kps2
    :param filePath: 文件路径
    :param separator: 数据分隔符，默认为一个tab
    :return: 读取到的坐标点数据list
    """
    kps1 = []
    kps2 = []
    open_file = open(filePath, 'r')
    num = open_file.readline().strip()
    data_line = open_file.readline().strip()
    while data_line:
        split_parts = data_line.split(separator)
        kps1.append((float(split_parts[0]), float(split_parts[1])))
        kps2.append((float(split_parts[2]), float(split_parts[3])))
        data_line = open_file.readline().strip()
    return kps1, kps2


def drawAndSaveMatches(img1, kps1, img2, kps2, save_path, color=[0, 0, 255], rad=5, thickness=1):
    """
    drawMatches()函数的二次封装，直接保存图片
    :param img1: 影像1
    :param kps1: 影像1上匹配的特征点
    :param img2: 影像2
    :param kps2: 影像2上匹配的特征点
    :param save_path: 输出影像的路径
    :param color: 特征点及连线的颜色，默认为红色，-1表示随机
    :param rad: 特征点大小，默认为5
    :param thickness: 匹配连线的宽度，默认为1
    :return: 无
    """
    match_img = drawMatches(img1, kps1, img2, kps2, color, rad, thickness)
    cv2.imwrite(save_path, match_img)


def findAffine(kps1, kps2):
    """
    基于匹配的特征点对求解仿射关系 -> affine
    :param kps1: 匹配的特征点1
    :param kps2: 匹配的特征点2
    :return: 仿射矩阵
    """

    if kps1.__len__() < 3 or kps2.__len__() < 3:
        affine = None
    else:
        if type(kps1[0]) is cv2.KeyPoint:
            tmp_kps1 = cvtCvKeypointToNormal(kps1)
        else:
            tmp_kps1 = kps1
        if type(kps2[0]) is cv2.KeyPoint:
            tmp_kps2 = cvtCvKeypointToNormal(kps2)
        else:
            tmp_kps2 = kps2
        affine, mask = cv2.estimateAffine2D(np.array(tmp_kps1), np.array(tmp_kps2))
    return affine


def findHomography(kps1, kps2):
    """
    基于匹配的特征点对求解单应变换关系 -> homo
    :param kps1: 匹配的特征点1
    :param kps2: 匹配的特征点2
    :return: 单应矩阵
    """

    if kps1.__len__() < 5 or kps2.__len__() < 5:
        homo = None
    else:
        if type(kps1[0]) is cv2.KeyPoint:
            tmp_kps1 = cvtCvKeypointToNormal(kps1)
        else:
            tmp_kps1 = kps1
        if type(kps2[0]) is cv2.KeyPoint:
            tmp_kps2 = cvtCvKeypointToNormal(kps2)
        else:
            tmp_kps2 = kps2
        homo, mask = cv2.findHomography(np.array(tmp_kps1), np.array(tmp_kps2))
    return homo


def checkAffineAccuarcy(kp1, kp2):
    """
    检查仿射模型精度 -> accuracy
    :param kp1: 影像1中的匹配点
    :param kp2: 影像2中的匹配点
    :return: 仿射变换的精度
    """

    affine = findAffine(kp1, kp2)
    T = np.mat(affine[:, 2].reshape(2, 1))
    R = np.mat(affine[:2, :2])

    kp2_ = []
    for i in range(kp1.__len__()):
        pt1 = np.mat(kp1[i]).reshape(2, 1)
        pt2_ = R * pt1 + T
        kp2_.append(pt2_)

    accuracy = []
    for i in range(kp2_.__len__()):
        dx = kp2_[i][0] - np.mat(kp2[i]).reshape(2, 1)[0]
        dy = kp2_[i][1] - np.mat(kp2[i]).reshape(2, 1)[1]
        d = np.sqrt(dx * dx + dy * dy)
        accuracy.append((float(dx), float(dy), float(d)))
    return accuracy


def resampleImg(img, trans):
    """
    基于获得的变换关系对影像进行重采 -> resampled_img
    :param img: 待重采影像
    :param trans: 变换矩阵，仿射or单应
    :return: 重采后的影像
    """

    if trans is None:
        return img
    if trans.shape[0] == 2:
        resampled_img = cv2.warpAffine(img, trans,
                                       (img.shape[1],
                                        img.shape[0]))
    elif trans.shape[0] == 3:
        resampled_img = cv2.warpPerspective(img, trans,
                                            (img.shape[1],
                                             img.shape[0]))
    else:
        resampled_img = img
    return resampled_img


def resampleToBase(img_base, img_warp, flag='affine'):
    """
    依据获得的变换关系，将影像重采到基准影像上 -> resample_img
    :param img_base: 基准影像路径
    :param img_warp: 待重采影像路径
    :param flag: 变换模型，'affine'(仿射)或'homo'(单应)
    :return: 重采后的影像
    """

    img1 = cv2.imread(img_base)
    img2 = cv2.imread(img_warp)
    kp1, des1 = getSiftKps(img1)
    kp2, des2 = getSiftKps(img2)
    good_kp1, good_kp2 = flannMatch(kp1, des1, kp2, des2)
    if flag == 'affine':
        trans = findAffine(good_kp2, good_kp1)
    elif flag == 'homo':
        trans = findHomography(good_kp2, good_kp1)
    else:
        trans = None
    resample_img = resampleImg(img2, trans)
    return resample_img


def getMinAndMaxGrayValue(img, ratio=0.0, bits=8):
    """
    获得影像中的灰度最大和最小值 -> min_gray, max_gray
    :param bits: 影像量化等级，默认8bit量化
    :param img: 待统计的影像
    :param ratio: 统计时首尾忽略像素的百分比占比，默认为0
    :return: 灰度最小、最大值
    """

    bins = np.arange(pow(2, bits))
    hist, bins = np.histogram(img, bins)
    total_pixels = img.shape[0] * img.shape[1]
    min_index = int(ratio * total_pixels)
    max_index = int((1 - ratio) * total_pixels)
    min_gray = 0
    max_gray = 0
    sum = 0
    for i in range(hist.__len__()):
        sum = sum + hist[i]
        if sum > min_index:
            min_gray = i
            break
    sum = 0
    for i in range(hist.__len__()):
        sum = sum + hist[i]
        if sum > max_index:
            max_gray = i
            break
    return min_gray, max_gray


def linearStretch(img, new_min, new_max, ratio=0.0):
    """
    灰度线性拉伸 -> img
    :param img: 待拉伸影像
    :param new_min: 期望最小值
    :param new_max: 期望最大值
    :param ratio: 拉伸百分比，默认为0，若为0.02则为2%线性拉伸
    :return: 拉伸后的影像
    """

    old_min, old_max = getMinAndMaxGrayValue(img, ratio)
    img1 = np.where(img < old_min, old_min, img)
    img2 = np.where(img1 > old_max, old_max, img1)
    print("=>linear stretch:")
    print('old min = %d,old max = %d new min = %d,new max = %d' % (old_min, old_max, new_min, new_max))
    img3 = np.uint8((new_max - new_min) / (old_max - old_min) * (img2 - old_min) + new_min)
    return img3
