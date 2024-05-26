from PIL import Image
import numpy as np
from tqdm import tqdm
from sympy import mod_inverse
from decimal import Decimal
import random

# 定义n：n个参与者
# 定义r：至少需要r个参与者恢复秘密
# 定义path：需要共享的图像数据
n = 5
r = 3
path = "test1.jpeg"

# 读取图像转换成np.array并展平
def read_image(path):
    img = Image.open(path).convert('RGB') 
    # img.show()
    img_array = np.asarray(img)
    # print(img_array.shape)
    return img_array.flatten(), img_array.shape

# 生成秘密子图
def polynomial(img, n, r):
    num_pixels = img.shape[0]
    # 随机生成多项式系数
    coef = np.random.randint(low = 0, high = 251, size = (num_pixels, r - 1))
    # print(coef.shape)

    gen_imgs = []
    # 循环n次生成n份秘密子图的一维数组形式
    for i in range(1, n + 1):
        base = np.array([i ** j for j in range(1, r)])
        # print(base)
        base = np.matmul(coef, base)
        # print(base.shape)
        # print(img.shape)
        img_ = img + base
        img_ = img_ % 251
        gen_imgs.append(img_)
    return np.array(gen_imgs)

def reconstruct_secret(shares):
    """
    利用拉格朗日插值法(已知m个秘密)还原并得到secret(f(0))
    """
    sums = 0
    # 拉格朗日插值公式
    for j, share_j in enumerate(shares):
        xj, yj = share_j
        prod = Decimal(1)

        for i, share_i in enumerate(shares):
            xi, _ = share_i
            if i != j:
                """
                重点：有模素数的多项式恢复要取模
                如果出现分数,需要计算分母的逆元,并同时乘分母与逆元(相当于乘1)
                调用mod_inversion函数计算逆元
                """
                #print(Decimal(Decimal(xi) / (xi - xj)))
                inverse = mod_inverse((xi-xj), 251)
                prod *= Decimal(((Decimal(xi) / (xi - xj))*(xi-xj)*inverse)%251)
        #print(yj)
        prod *= yj
        sums += Decimal(prod)
    # print(sums)

    return int(round(Decimal(sums), 0))


    
if __name__ == "__main__":
    img_flattened, shape = read_image(path)

    # 使用exceed列表存储大于250像素值与250的差值
    exceed = []
    for i in range(len(img_flattened)):
        if img_flattened[i]>=250:
            exceed.append(img_flattened[i]-250)

    # 将大于250的像素值改为250,（防止模251而产生损失，配合exceed列表实现无损）
    img_flattened[img_flattened>=250] = 250

    gen_imgs = polynomial(img_flattened, n = n, r = r)
    # reshape恢复成图像原来的大小，保存分发的影子图像
    to_save = gen_imgs.reshape(n, *shape)
    for i, img in enumerate(to_save):
        Image.fromarray(img.astype(np.uint8)).save("distribute_{}.png".format(i + 1))
    
    distribute_img = []

    # img1,_ = read_image("distribute_1.png")
    # t = (1,img1)
    # distribute_img.append(t)

    # img2,_ = read_image("distribute_2.png")
    # t = (2,img2)
    # distribute_img.append(t)

    # img3,_ = read_image("distribute_3.png")
    # t = (3,img3)
    # distribute_img.append(t)

    # img4,_ = read_image("distribute_4.png")
    # distribute_img.append((4,img4))

    # img5,_ = read_image("distribute_5.png")
    # distribute_img.append((5,img5))


    # 读取分发的影子图像
    for i in range(n):
        img,_ = read_image("distribute_{}.png".format(i+1))
        distribute_img.append((i+1,img))

    # 随机选取r个影子图像的数据用作恢复图像
    decode_img = random.sample(distribute_img, r)

    # 恢复图像阶段
    dim = img_flattened.shape[0]
    recover_img = []

    # 逐像素调用reconstruct_secret函数恢复秘密像素值
    # tqdm可视化进度
    for i in tqdm(range(dim)):
        secr = []
        for _,shares in enumerate(distribute_img):
            temp = shares[1][i]
            t = (shares[0],temp)
            secr.append(t)
        recover_img.append(reconstruct_secret(secr)%251)

    # 按顺序遍历恢复出来的一维秘密像素，如果恢复的像素等于250，则按照之前记录的
    # exceed列表恢复真实的像素值
    j = 0
    for i in range(len(recover_img)):
        if recover_img[i]==250:
            recover_img[i] += exceed[j]
            j += 1

    # reshape成原来的形状并保存
    recover_img = np.array(recover_img)
    recover_img = recover_img.reshape(*shape)
    Image.fromarray(recover_img.astype(np.uint8)).save("recover_img.png")


    # 逐像素点判断恢复的图像是否与原图像一致，若不一致，则定位到位置并打印像素点
    img1 = Image.open(path)
    img2 = Image.open("recover_img.png")

    pixels1 = list(img1.getdata())
    pixels2 = list(img2.getdata())

    differences = []

    for i, (p1, p2) in enumerate(zip(pixels1, pixels2)):
        if p1 != p2:
            differences.append((i % img1.width, i // img1.width))

    if differences:
        print("发现不一致的像素点：")
        for diff in differences:
            print(f"位置：({diff[0]}, {diff[1]})，像素值：{pixels1[diff[1] * img1.width + diff[0]]} != {pixels2[diff[1] * img2.width + diff[0]]}")
    else:
        print("两个图像的像素完全一致。")