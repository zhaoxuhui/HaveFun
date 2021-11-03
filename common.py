# coding=utf-8
import os
import numpy as np


def findFiles(root_dir, filter_type, reverse=False):
    """
    在指定目录查找指定类型文件 -> paths, names, files
    :param root_dir: 查找目录
    :param filter_type: 文件类型
    :param reverse: 是否返回倒序文件列表，默认为False
    :return: 路径、名称、文件全路径
    """

    separator = os.path.sep
    paths = []
    names = []
    files = []
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter_type):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print(names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    if reverse:
        paths.reverse()
        names.reverse()
        files.reverse()
    return paths, names, files


def isDirExist(path='output'):
    """
    判断指定目录是否存在，如果存在返回True，否则返回False并新建目录 -> bool
    :param path: 指定目录
    :return: 判断结果
    """

    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True


def getRandomNum(start=0, end=100):
    """
    获取指定范围内的随机整数，默认范围为0-100 -> rand_num
    :param start: 最小值
    :param end: 最大值
    :return: 随机数
    """

    return np.random.randint(start, end + 1)
