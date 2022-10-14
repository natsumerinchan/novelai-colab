# [有手就会系列] 四步教你用NovelAI生成二次元小姐姐

### 原作者:forever豪3 [点击访问原文](https://aistudio.baidu.com/aistudio/projectdetail/4666819)

**介于飞桨AIStudio太垃圾,故迁移到Colab**

**[点击这里前往END(有讨论群二维码)](#end)**

## 第一步:运行代码!
进入之后，点击框里左上角的“[ ]”运行
下面的代码只用在你**第一次进入**时运行！
下面的代码里在云环境里解压了模型文件，并且自动安装了各个文件(但是贴心的我已经为您解压好了)
  ```
import os
from IPython.display import clear_output
from utils import check_is_model_complete, model_unzip, diffusers_auto_update
# 如果需要生成 768 x 1024 则把下面一行 False 改成 True, 但是可能卡, 建议挑人少的时候
diffusers_auto_update(dev_paddle = False)
clear_output() # 清理很长的内容
print('加载完毕, 请重启内核')
```
## 第二步:重启内核!
![](https://raw.githubusercontent.com/Wangs-offical/PictureBed-Wangs/master/2022/10/14/AwgTzHYUsYuHsu9c.png)
## 第三步:开玩!
点击左上角的“[ ]”运行下面的代码，等几秒加载模型就可以玩耍啦~ **以后每次进来直接运行下面这个就行啦~**

   ```
from ui import gui_txt2img # 文生图
display(gui_txt2img.gui) # 生成的图片自动保存到左侧的 outputs 的文件夹里
```

   ```
from ui import gui_img2img # 图生图, 在左侧上传图片, 然后修改 "需要转换的图片路径"
display(gui_img2img.gui) # 生成的图片自动保存到左侧的 outputs 的文件夹里
```

   ```
from ui import gui_superres # 超分 (图片放大一倍), 在左侧上传图片, 然后修改 "需要超分的图片路径"
display(gui_superres.gui) # 生成的图片自动保存到左侧的 outputs 的文件夹里
```

**以下是直接从原文搬的**

## 常见问题


> **Q**: 能不能用中文描述啊？
> 
> **A**: 可以的, 但是效果不如英文。因为模型是用英语文本训练的。（可以去翻译软件里翻译一下）

> **Q**: 生成的图怎么都一样的？
> 
> **A**: 如果设定了相同的随机数种子且其它参数相同，则生成的图也是一样的。

> **Q**: 生成的时候进度条卡住， 然后弹出一个框框：The kernel for main.ipynb appears to have died. It will restart automatically.？
> 
> **A**: 生成图片尺寸太大了！可以重启内核改用较小的尺寸，或者重新进入更换更大的 GPU。

> **Q**: 其它神秘问题？
> 
> **A**: 请去评论区进讨论群。如果是默认的参数无法生成，可以删除该项目重新 Fork 一份从头开始。


# 非界面版

如果你不满足于上面的界面玩法, 那么不妨来试试下面的部分！（下面有图生图哦）

### 如果第一次运行下面报错, 不要惊慌！重启内核重新运行一次！


```
# 如果第一次运行下面报错, 就重启内核重新运行一次
from diffusers_paddle import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import paddle
from utils import save_image_info, model_vae_get_default, model_get_default
import os
pipe = StableDiffusionPipeline.from_pretrained(os.path.join('./',os.path.basename(model_get_default()).rstrip('.zip')))

vae_path = os.path.join('./',os.path.basename(model_get_default()).rstrip('.zip'),'vae',os.path.basename(model_vae_get_default()))
vae_path = vae_path if os.path.exists(vae_path) else model_vae_get_default() # 加载 vae
pipe.vae.load_state_dict(paddle.load(vae_path)) # 换用更好的 vae (有效果!)

# 图生图
pipe_i2i = StableDiffusionImg2ImgPipeline(vae=pipe.vae,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,
unet=pipe.unet,scheduler=pipe.scheduler,safety_checker=pipe.safety_checker,feature_extractor=pipe.feature_extractor)
print('加载完毕')
```

# 文生图


参数:

-   height 图片高度 （常用 512,768,1024等）
-   width 图片宽度 （常用 512,768,1024等）
-   seed 随机数种子（如果是None则为自动随机）
-   steps 生成步数
-   cfg 引导比例
-   **prompt 描述内容**
-   negative_prompt 反面描述内容

注：生成图片尺寸越大, 则需要时间越久

注: 描述内容太长的话, 可以用右斜线 \ 换行.

生成的图都在 outputs 文件夹里。


```
height = 512
width  = 768
seed   = None
steps  = 50
cfg    = 7    
prompt = "miku, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, \
cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"


image = pipe(prompt,height=height,width=width,num_inference_steps=steps,guidance_scale=cfg,seed=seed,negative_prompt=negative_prompt).images[0]
print('Seed =', image.argument['seed'])
display(image)
save_image_info(image, path = './outputs/') # 保存图片到指定位置
```
<div id="end">

# End

![285925041 点击查看二维码](https://ai-studio-static-online.cdn.bcebos.com/5f439c18f5764a74a1bd5702de893a89252ff85130844d83a5e92da1d180f10d)

作者的讨论群

**如果你认为这个Colab版本对你有帮助的话,[欢迎点亮star!](https://github.com/Wangs-offical/novelai-colab/stargazers)**

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTY2Njc1NTAzNCwtMTc5MTU3Mjc0MV19
-->