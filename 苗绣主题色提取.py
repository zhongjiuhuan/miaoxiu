import os
import torch
from kmeans_pytorch import kmeans
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as color
import webcolors
import json
import cv2
import math


def get_image(folder_path):
    '''
    获取目录路径下所有的图片
    Args:
        folder_path:图片所在的路径

    Returns:
        image_paths:当前目录下所有图片的路径
        image_names:图片文件的名称

    '''
    assert os.path.isdir(folder_path), '{:s} is not a valid directory'.format(folder_path)
    img_extensions = ['.jpg', '.JPG', '.jpeg', '.png', '.PNG', '.ppm', '.bmp', '.BMP', '.tif']
    # 获取图片的路径
    images = []
    image_names = []
    for fname in sorted(os.listdir(folder_path)):
        if any(fname.endswith(extension) for extension in img_extensions):
            imag_path = os.path.join(folder_path, fname)
            images.append(imag_path)
            image_names.append(fname)
    image_paths = sorted(images)
    return image_paths, image_names


def read_images(image_path, height=256, width=256):
    '''
    根据路径读取文件，并将其裁剪为固定大小的图片
    Args:
        image_path: 图片的路径
        height: 需要裁剪的图片高度大小
        width: 需要裁剪图片宽度大小

    Returns:
        hsv_img: hsv格式的图片矩阵
        rgb_img: rgb格式的图片矩阵

    '''
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h = img.shape[0]
    w = img.shape[1]
    if height > h:
        height = h
    if width > w:
        width = w
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # h:0~180, s:0~255, v:0~255
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return hsv_img, rgb_img


def hsv2rgb(h, s, v):
    '''
    将hsv格式转换成rgb
    '''
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def hsv2hex(hsv):
    '''
    将RGB转换成hex格式
    Args:
        hsv: hsv格式颜色

    Returns:
        hex: hsv格式颜色对应的rgb颜色
    '''
    h, s, v = hsv
    h = h * 2
    s = s / 255
    v = v / 255
    R, G, B = hsv2rgb(h, s, v)  #
    hex = color.to_hex([int(R) / 255, int(G) / 255, int(B) / 255])

    return hex


def find_color_name(hsv):
    '''
    Finding color name :: returning hex code and nearest/actual color name
    '''
    h, s, v = hsv
    h = h * 2
    s = s / 255
    v = v / 255
    R, G, B = hsv2rgb(h, s, v)
    aname, cname = get_colour_name((int(R), int(G), int(B)))
    hex = color.to_hex([int(R) / 255, int(G) / 255, int(B) / 255])
    if aname is None:
        name = cname
    else:
        name = aname
    return hex, name


def closest_colour(requested_colour):
    '''
    We are basically calculating euclidean distance between our set of RGB values
    with all the RGB values that are present in our JSON. After that, we are looking
    at the combination RGB (from JSON) that is at least distance from input
    RGB values, hence finding the closest color name.
    '''
    min_colors = {}
    for key, name in color_dict['color_names'].items():
        r_c, g_c, b_c = webcolors.hex_to_rgb("#" + key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colors[math.sqrt(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_colour_name(requested_colour):
    '''
    In this function, we are converting our RGB set to color name using a third
    party module "webcolors".

    RGB set -> Hex Code -> Color Name

    By default, it looks in CSS3 colors list (which is the best). If it cannot find
    hex code in CSS3 colors list, it raises a ValueError which we are handling
    using our own function in which we are finding the closest color to the input
    RGB set.
    '''
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def plot_color_clusters(cluster_map, rgb_img=None, plt_save_dir=None, imshow=False):
    # grouping the data by color hex code and color name to find the total count of
    # pixels (data points) in a particular cluster
    mydf = cluster_map.groupby(['color', 'color_name']).agg({'position': 'count'}).reset_index().rename(
        columns={"position": "count"})
    #     mydf['H'] = [i[0] for i in cluster_center.tolist()]
    #     mydf['S'] = [i[1]/255*100 for i in cluster_center.tolist()]
    #     mydf['V'] = [i[2]/255*100 for i in cluster_center.tolist()]
    mydf['Percentage'] = round((mydf['count'] / mydf['count'].sum()) * 100, 1)
    print(mydf)

    #     '''
    #     Subplots with image and a pie chart representing the share of each color identified
    #     in the entire photograph/image.
    #     '''
    if rgb_img is not None:
        plt.figure(figsize=(7, 4))
        plt.subplot(121)
        plt.imshow(rgb_img)
        plt.axis('off')
        plt.subplot(122)
        plt.pie(mydf['count'], labels=mydf['color_name'], colors=mydf['color'], autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
    else:
        plt.pie(mydf['count'], labels=mydf['color_name'], colors=mydf['color'], autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
    if imshow:
        plt.show()
    if plt_save_dir is not None:
        plt.savefig(plt_save_dir)
    plt.close('all')


def hsv_main_color_extraction(image_vector, k):
    '''
    hsv格式图片主题色提取函数
    Args:
        image_vector: hsv格式图片的向量(w*h, 3)
        k: 需要聚的类个数

    Returns:
        cluster_map: 记录了颜色和颜色名等信息，用于后面的绘图
        cluster_centers: 聚类的中心点

    '''
    cluster_ids_x, cluster_centers = kmeans(
        X=image_vector, num_clusters=k, distance='euclidean', device=torch.device('cuda:0'))
    # 将center的颜色转换成HEX格式
    hex_colors = [hsv2hex(hsv.tolist()) for hsv in cluster_centers]
    color_name = {}
    for c in cluster_centers:
        h, name = find_color_name(c.tolist())
        color_name[h] = name
    img_cor = [[*x] for x in image_vector]
    '''
    img_cor is a nested list of all the coordinates (pixel -- RGB value) present in the
    image
    '''
    cluster_map = pd.DataFrame()
    cluster_map['position'] = img_cor
    cluster_map['cluster'] = cluster_ids_x
    cluster_map['color'] = [hex_colors[x] for x in cluster_map['cluster']]
    cluster_map['color_name'] = [color_name[x] for x in cluster_map['color']]
    return cluster_map, cluster_centers


def main():
    # 开始第一次聚类
    image_paths, image_names = get_image(folder_path)
    all_centers = []
    # 读取图片
    for i, image_path in enumerate(image_paths):
        image_name = image_names[i]
        hsv_img, rgb_img = read_images(image_path, img_height, img_width)
        # 对hsv图片进行聚类
        image_vector = torch.from_numpy(hsv_img.reshape((-1, 3)))
        cluster_map, cluster_centers = hsv_main_color_extraction(image_vector, 5)
        for center in cluster_centers.tolist():
            all_centers.append(center)
        print("文件名为:{0}".format(image_name))
        # 绘制图片信息
        plot_color_clusters(cluster_map, rgb_img, os.path.join(plt_save_dir, image_name))
    # 第二次聚类
    all_centers = torch.Tensor(all_centers)
    cluster_map, _ = hsv_main_color_extraction(all_centers, 10)
    # 绘制图片信息
    print("{0}的主题色为：".format(file_name))
    plot_color_clusters(cluster_map, plt_save_dir=os.path.join(plt_save_dir, 'result.jpg'))


if __name__ == '__main__':
    # 加载colors.json文件，用于显示颜色名称
    with open(r'/home/wusong/sdb/lpj/notebooks/colors.json') as clr:
        color_dict = json.load(clr)
    # 需要计算的图片所在目录
    folder_path = r'/home/wusong/sdb/lpj/苗绣/2020-5-数纱绣'
    # 图片最大长宽
    img_width = 256
    img_height = 256
    file_name = '2020-5-数纱绣'
    # 结果保存路径
    plt_save_dir = r'/home/wusong/sdb/lpj/苗绣/result/2020-5-数纱绣'
    # 检查保存路径是否存在，若不就创建路径
    if not os.path.isdir(plt_save_dir):
        os.makedirs(plt_save_dir)
    main()
