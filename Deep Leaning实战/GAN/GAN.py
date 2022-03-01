


"""
本程序使用Generative Adversarial Network(GAN)实现二次元头像的生成

Generator解析: 
    1. 一个神经网络或者一个函数
    2. 向生成器中输人一个向量, 就可以输出一些东西。
    3. 输人一个向量, 生成器便会生成一张图片。通常, 输人向量的每一个维度都会对应图片的某一种特征。
    ┌─┐         ┌───────────┐
    | |         |           |         ┌─────┐
    | |────────▶| Generator |────────▶| IMG |
    | |         |           |         └─────┘
    └─┘         └───────────┘        
    向量

Discriminator解析:
    1. 用于训练生成器
    2. 一个神经网络或者一个函数
    3. 输出为标量, 0~1, 接近1表示图片或数据越真实
                    ┌───────────┐
    ┌─────┐         |           |         
    | IMG |────────▶|  Discrim  |────────▶标量
    └─────┘         |           |
                    └───────────┘

生成对抗网络训练逻辑:
        ┌──────────┐          ┌──────────┐           ┌──────────┐
        |          |          |          |           |          |
        | Gene 1.0 |─────────▶| Gene 2.0 |──────────▶| Gene 3.0 |
        |          |          |          |           |          |
        └──────────┘          └──────────┘           └──────────┘
             |                      |                      |
             |                      |                      |
             ▼                      ▼                      ▼
    ┌───┐───┐───┐───┐───┐  ┌───┐───┐───┐───┐───┐  ┌───┐───┐───┐───┐───┐
    |   |   |   |   |   |  |   |   |   |   |   |  |   |   |   |   |   |
    └───┘───┘───┘───┘───┘  └───┘───┘───┘───┘───┘  └───┘───┘───┘───┘───┘
          生成的图片               生成的图片              生成的图片
             |                      |                      |
             |                      |                      |
             ▼                      ▼                      ▼
        ┌──────────┐          ┌──────────┐           ┌──────────┐
        |          |          |          |           |          |
        | Disc 1.0 |─────────▶| Disc 2.0 |──────────▶| Disc 3.0 |
        |          |          |          |           |          |
        └──────────┘          └──────────┘           └──────────┘
             ▲                      ▲                      ▲
             |                      |                      |
             |                      |                      |
             |             ┌───┐───┐───┐───┐───┐           |
             └─────────────|   |   |   |   |   |───────────┘
                           └───┘───┘───┘───┘───┘
                                  真实图片

GAN算法流程:
    1. 初始化生成器和鉴别器的参数
    2. 每次训练迭代中进行如下操作:
        a. 固定生成器, 升级鉴别器
        b. 固定鉴别器, 升级生成器

固定生成器升级鉴别器图解:
                                                                                   Self update
                                                                               ┌──────────────────┐
                                                                               |                  |
                                        True Pictures                          ▼                  |
┌───┐───┐───┐───┐───┐    Sample     ┌───┐───┐───┐───┐───┐               ┌─────────────┐           |
|   |   |   |   |   |──────────────▶|   |   |   |   |   |──────────────▶|             |           |
└───┘───┘───┘───┘───┘               └───┘───┘───┘───┘───┘               |   Discrim   |───────────┘
    True Data                       ┌───┐───┐───┐───┐───┐               |     3.0     | 
                                    |   |   |   |   |   |──────────────▶|             |
                                    └───┘───┘───┘───┘───┘               └─────────────┘
                                        Gene Pictures
                                              ▲
                                              |
                                              |
                ┌─┐┌─┐┌─┐                ┌──────────┐
                | || || |                |          |
                | || || |──────────────▶ |   Gene   |
                | || || |                |          |
                | || || |                └──────────┘
                └─┘└─┘└─┘                    Fixed 
              Random Vectors     
"""