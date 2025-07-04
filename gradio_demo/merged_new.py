import sys
sys.path.append('./')

# 创建huggingface_hub兼容性补丁
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download

from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np  
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import subprocess
import platform
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import shutil
import time
import requests
import json
import base64
import io

# 添加服务器配置
SERVER_URL = "http://localhost:8080"  # 根据实际情况修改服务器地址
current_user_info = {}  # 存储当前用户信息
user_session = requests.Session()  # 用于保持会话

# 自定义保存路径配置
DEFAULT_SAVE_PATH = "/root/autodl-tmp/IDM-VTON/gradio_demo/example/result"
allowed_paths = [
    "D:\\WorkSpace\\code\\my_idm_vton\\local_client\\saved_images",  # API返回的用户文件路径
    DEFAULT_SAVE_PATH,  # 本地保存路径
]

# 设备配置：优先使用GPU，否则使用CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

# 模型路径配置
base_path = 'D:\Work\IDM_VTON\cache'
example_path = os.path.join(os.path.dirname(__file__), 'example')

# 加载所有模型（保持tryapp.py的模型加载代码）
unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
    local_files_only=True
)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
    local_files_only=True 
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
    local_files_only=True 
)

noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler",local_files_only=True )

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
    local_files_only=True 
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
    local_files_only=True 
)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    local_files_only=True 
    )

vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
                                    local_files_only=True 
)

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
    local_files_only=True 
)

# # 初始化人体解析和姿态检测模型
parsing_model = Parsing(0)
openpose_model = OpenPose(0)

# # 冻结所有模型参数
UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

# # 定义图像预处理变换
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

# # 创建试穿管道
pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        local_files_only=True 
)
pipe.unet_encoder = UNet_Encoder

def start_tryon(dict,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = dict["background"].convert("RGB")    
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
    
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray

# 保持tryapp.py的所有辅助函数

def select_human_image():
    """
    选择人像图片文件
    
    返回:
        选择的图片和状态信息
    """
    try:
        # 使用默认的初始目录
        human_folder = os.path.join(example_path, "human")
        initial_dir = human_folder if os.path.exists(human_folder) else os.path.dirname(__file__)
        
        # 打开文件选择对话框
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="选择人像图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("PNG文件", "*.png"),
                ("所有文件", "*.*")
            ]
        )
        
        root.destroy()  # 销毁tkinter窗口
        
        if file_path:
            # 构建ImageEditor需要的字典格式
            image_dict = {
                'background': file_path,
                'layers': None,
                'composite': None
            }
            return image_dict, f"已选择人像图片: {os.path.basename(file_path)}"
        else:
            return None, "未选择文件"
    
    except Exception as e:
        return None, f"选择文件失败: {str(e)}"

def select_garment_image():
    """
    选择服装图片文件
    
    返回:
        选择的图片和状态信息
    """
    try:
        # 使用默认的初始目录
        cloth_folder = os.path.join(example_path, "cloth")
        initial_dir = cloth_folder if os.path.exists(cloth_folder) else os.path.dirname(__file__)
        
        # 打开文件选择对话框
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="选择服装图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("PNG文件", "*.png"),
                ("所有文件", "*.*")
            ]
        )
        
        root.destroy()  # 销毁tkinter窗口
        
        if file_path:
            image = Image.open(file_path).convert('RGB')
            return image, f"已选择服装图片: {os.path.basename(file_path)}"
        else:
            return None, "未选择文件"
    
    except Exception as e:
        return None, f"选择文件失败: {str(e)}"

def save_generated_image(result_image, mask_image, filename_prefix="tryon_result"):
    """
    保存生成的图片到预设路径
    
    参数:
        result_image: 试穿结果图像
        mask_image: 掩码图像（不再使用，保留参数以保持兼容性）
        filename_prefix: 文件名前缀
    
    返回:
        保存状态信息
    """
    try:
        # 使用预设的保存路径
        save_path = DEFAULT_SAVE_PATH
        
        # 创建保存目录（如果不存在）
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        
        # 只保存试穿结果图像
        if result_image is not None:
            # 检查并转换图像类型
            if isinstance(result_image, np.ndarray):
                # 如果是numpy数组，转换为PIL Image
                result_image = Image.fromarray(result_image.astype(np.uint8))
            elif hasattr(result_image, 'numpy'):
                # 如果是tensor，先转换为numpy再转换为PIL
                result_image = Image.fromarray(result_image.numpy().astype(np.uint8))
            
            result_filename = f"{filename_prefix}_result_{timestamp}.png"
            result_path = os.path.join(save_path, result_filename)
            result_image.save(result_path)
            saved_files.append(result_filename)
        
        if saved_files:
            return f"✅ 图片已成功保存到: {save_path}\n📁 保存的文件: {', '.join(saved_files)}"
        else:
            return "❌ 没有可保存的图片，请先生成试穿结果"
    
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"

def save_to_collection(result_image, filename_prefix="collection"):
    """
    将图片保存到收藏路径
    
    参数:
        result_image: 试穿结果图像
        filename_prefix: 文件名前缀
    
    返回:
        保存状态信息和保存的文件路径
    """
    try:
        # 使用预设的保存路径
        save_path = DEFAULT_SAVE_PATH
        
        # 创建保存目录（如果不存在）
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if result_image is not None:
            # 检查并转换图像类型
            if isinstance(result_image, np.ndarray):
                result_image = Image.fromarray(result_image.astype(np.uint8))
            elif hasattr(result_image, 'numpy'):
                result_image = Image.fromarray(result_image.numpy().astype(np.uint8))
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"{filename_prefix}_{timestamp}.png"
            result_path = os.path.join(save_path, result_filename)
            result_image.save(result_path)
            
            return f"✅ 收藏成功！图片已保存到: {result_filename}", result_path
        else:
            return "❌ 没有可收藏的图片，请先生成试穿结果", None
    
    except Exception as e:
        return f"❌ 收藏失败: {str(e)}", None

def load_collection_images():
    """
    从收藏路径加载所有图片
    
    返回:
        图片路径列表和状态信息
    """
    try:
        save_path = DEFAULT_SAVE_PATH
        
        if not os.path.exists(save_path):
            return [], " 收藏文件夹不存在"
        
        # 获取文件夹中的所有图片文件
        files, status = get_folder_files(save_path)
        
        if files:
            # 按修改时间倒序排列（最新的在前面）
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return files, f"✅ 加载了 {len(files)} 张收藏图片"
        else:
            return [], " 收藏文件夹为空"
            
    except Exception as e:
        return [], f"❌ 加载收藏失败: {str(e)}"

def select_from_human_gallery(evt: gr.SelectData):
    try:
        current_files, _ = browse_human_folder()
        if evt.index < len(current_files):
            selected_image_path = current_files[evt.index]['background']
            image = Image.open(selected_image_path).convert('RGB')
            image_dict = {
                'background': image,
                'layers': None,
                'composite': None
            }
            return image_dict, f"✅ 已选择人像图片: {os.path.basename(selected_image_path)}"
        else:
            return None, "❌ 选择的图片索引超出范围"
    except Exception as e:
        return None, f"❌ 选择图片失败: {str(e)}"

def select_from_cloth_gallery(evt: gr.SelectData):
    try:
        current_files, _ = browse_cloth_folder()
        if evt.index < len(current_files):
            selected_image_path = current_files[evt.index]
            image = Image.open(selected_image_path).convert('RGB')
            return image, f"✅ 已选择服装图片: {os.path.basename(selected_image_path)}"
        else:
            return None, "❌ 选择的图片索引超出范围"
    except Exception as e:
        return None, f"❌ 选择图片失败: {str(e)}"

def update_human_gallery():
    files, status = browse_human_folder()
    image_paths = [item['background'] for item in files] if files else []
    return gr.Gallery(value=image_paths, visible=len(image_paths) > 0), status

def update_cloth_gallery():
    files, status = browse_cloth_folder()
    return gr.Gallery(value=files, visible=len(files) > 0), status
def get_folder_files(folder_path, file_extensions=None):
    try:
        if not os.path.exists(folder_path):
            return [], f"❌ 文件夹不存在: {folder_path}"
        
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        
        files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in file_extensions:
                    files.append(file_path)
        
        files.sort()
        
        if files:
            return files, f"✅ 找到 {len(files)} 个图片文件"
        else:
            return [], f"⚠️ 文件夹中没有找到图片文件"
            
    except Exception as e:
        return [], f"❌ 读取文件夹失败: {str(e)}"

def browse_human_folder():
    """
    浏览人像图片文件夹，支持用户专属目录
    
    返回:
        文件列表和状态信息
    """
    try:
        # 如果用户已登录，获取用户专属的char目录
        if current_user_info.get('user_id'):
            files, status = get_user_file_paths('char')
        else:
            # 未登录用户使用默认示例文件夹
            folder_path = os.path.join(example_path, "human")
            if not os.path.exists(folder_path):
                folder_path = os.path.dirname(__file__)
            files, status = get_folder_files(folder_path)
        
        # 转换为人像示例字典格式
        human_list = []
        for file_path in files:
            ex_dict = {
                'background': file_path,
                'layers': None,
                'composite': None
            }
            human_list.append(ex_dict)
        
        return human_list, status
        
    except Exception as e:
        return [], f"❌ 浏览人像文件夹失败: {str(e)}"

def browse_cloth_folder():
    """
    浏览服装图片文件夹，支持用户专属目录
    
    返回:
        文件列表和状态信息
    """
    try:
        # 如果用户已登录，获取用户专属的clothes目录
        if current_user_info.get('user_id'):
            files, status = get_user_file_paths('clothes')
        else:
            # 未登录用户使用默认示例文件夹
            folder_path = os.path.join(example_path, "cloth")
            if not os.path.exists(folder_path):
                folder_path = os.path.dirname(__file__)
            files, status = get_folder_files(folder_path)
        
        return files, status
        
    except Exception as e:
        return [], f"❌ 浏览服装文件夹失败: {str(e)}"

def external_register_api(username, password, nickname):
    """
    调用服务器注册接口
    
    参数:
        username: 用户名
        password: 密码  
        nickname: 昵称(作为email使用)
    
    返回:
        bool: 注册是否成功
    """
    try:
        # 构造注册请求数据
        register_data = {
            "username": username,
            "email": f"{nickname}@local.com",  # 使用昵称构造邮箱
            "password": password
        }
        
        # 发送注册请求
        response = user_session.post(
            f"{SERVER_URL}/api/register",
            json=register_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('success', False)
        else:
            print(f"注册失败，状态码: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"注册请求失败: {str(e)}")
        return False

def external_login_api(username, password):
    """
    调用服务器登录接口
    
    参数:
        username: 用户名
        password: 密码
    
    返回:
        bool: 登录是否成功
    """
    global current_user_info
    
    try:
        # 构造登录请求数据
        login_data = {
            "username": username,
            "password": password,
            "remember_me": True
        }
        
        # 发送登录请求
        response = user_session.post(
            f"{SERVER_URL}/api/login",
            json=login_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success', False):
                # 保存用户信息和session cookies
                current_user_info = result.get('user', {})
                print(f"登录成功，用户ID: {current_user_info.get('user_id')}")
                return True
        else:
            print(f"登录失败，状态码: {response.status_code}")
            
        return False
        
    except Exception as e:
        print(f"登录请求失败: {str(e)}")
        return False

def get_user_file_paths(category='all'):
    """
    获取当前用户的文件路径信息
    
    参数:
        category: 分类，'all', 'clothes', 'char'
    
    返回:
        tuple: (文件路径列表, 状态信息)
    """
    try:
        if not current_user_info.get('user_id'):
            # 未登录用户，返回默认示例路径
            return [], "⚠️ 未登录，显示默认示例图片"
        
        # 发送获取文件路径请求
        response = user_session.get(
            f"{SERVER_URL}/api/user/file-paths",
            params={'category': category},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success', False):
                files = []
                paths_info = result.get('paths', {})
                
                # 根据分类提取文件路径
                if category == 'all' or category in paths_info:
                    categories_to_check = [category] if category != 'all' else paths_info.keys()
                    
                    for cat in categories_to_check:
                        if cat in paths_info:
                            cat_files = paths_info[cat].get('files', [])
                            files.extend([file_info['full_path'] for file_info in cat_files])
                
                stats = result.get('statistics', {})
                status = f"✅ 找到 {stats.get('total_files', 0)} 个文件 ({stats.get('total_size_mb', 0):.1f}MB)"
                
                return files, status
            else:
                return [], f"❌ 获取文件路径失败: {result.get('error', '未知错误')}"
        else:
            return [], f"❌ 服务器响应错误: {response.status_code}"
            
    except Exception as e:
        print(f"获取用户文件路径失败: {str(e)}")
        return [], f"❌ 获取文件路径失败: {str(e)}"

def is_logged_in(state):
    """
    检查登录状态
    
    参数:
        state: 状态变量
    
    返回:
        bool: 是否已登录
    """
    return bool(state) and bool(current_user_info.get('user_id'))


custom_css = """
body {
    background:
        linear-gradient(135deg, #e0e7ef 0%, #f8fafc 100%),
        radial-gradient(circle, #dbeafe 1px, transparent 1px),
        radial-gradient(circle, #dbeafe 1px, transparent 1px);
    background-size: 100%, 40px 40px, 40px 40px;
    background-position: 0 0, 0 0, 20px 20px;
}
#global-bg-img {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
    background: url('https://raw.githubusercontent.com/Shallow536/image/main/bg1.png') no-repeat center center;
    background-size: cover;
}
#global-bg-img::after {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: 0;
    background: rgba(255,255,255,0.3);
    pointer-events: none;
}
.gradio-container {
    position: relative;
    z-index: 1;
}
#fav-btn, #login-btn, #register-btn, #mode1-btn,
#login-btn2, #back-btn-login, #register-confirm-btn, #back-btn-register,
#try-btn, #fav-this-btn, #back-btn1,
#del-btn, #back-btn-fav,
#browse-human, #select-human, #browse-cloth, #select-cloth, #save-img-btn {
    font-size: 20px !important;
    border-radius: 16px !important;
    padding: 10px 24px !important;
    border: 2px solid #a5b4fc !important;
    background: linear-gradient(90deg, #f1f5fa 60%, #dbeafe 100%) !important;
    color: #2563eb !important;
    box-shadow: 0 2px 8px #e0e7ef;
    transition: background 0.2s, box-shadow 0.2s, transform 0.15s;
    opacity: 1 !important;
    display: inline-block !important;
    visibility: visible !important;
    z-index: 10 !important;
}
#fav-btn:hover, #login-btn:hover, #register-btn:hover, #mode1-btn:hover,
#login-btn2:hover, #back-btn-login:hover, #register-confirm-btn:hover, #back-btn-register:hover,
#try-btn:hover, #fav-this-btn:hover, #back-btn1:hover,
#del-btn:hover, #back-btn-fav:hover,
#browse-human:hover, #select-human:hover, #browse-cloth:hover, #select-cloth:hover, #save-img-btn:hover {
    background: linear-gradient(90deg, #dbeafe 60%, #f1f5fa 100%) !important;
    box-shadow: 0 6px 24px #a5b4fc;
    transform: translateY(-2px) scale(1.04);
    opacity: 1 !important;
    visibility: visible !important;
}
#fav-btn {
    position: absolute;
    top: 80px;
    left: 40px;
    z-index: 10;
}
#login-btn {
    position: absolute;
    top: 20px;
    right: 220px;
    z-index: 10;
}
#register-btn {
    position: absolute;
    top: 20px;
    right: 40px;
    z-index: 10;
}
#center-btn-row {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 120px 0 40px 0;
    overflow: visible !important;  /* 新增，允许子元素放大时不被裁剪 */
    overflow: visible !important;
    width: auto !important;
    max-width: none !important;
}


#back-btn1, button##back-btn1, div##back-btn1 {
    width: 600px !important;      /* 你想要的宽度 */
    flex: none !important;        /* 防止被拉伸 */
    max-width: 750px !important;  /* 防止超出 */
    min-width: 600px !important;
}
#back-btn1 {
    z-index: 100 !important;
    position: relative !important;
    width: 700px !important;
    height: 50px !important;
    font-size: 20px !important;
    margin: 0 auto !important;
    border-radius: 22px !important;
    display: flex;
    justify-content: center;
    align-items: center;
}

#mode1-btn, button#mode1-btn, div#mode1-btn {
    width: 550px !important;      /* 你想要的宽度 */
    flex: none !important;        /* 防止被拉伸 */
    max-width: 600px !important;  /* 防止超出 */
    min-width: 550px !important;
}
#mode1-btn {
    z-index: 100 !important;
    position: relative !important;
    width: 600px !important;
    height: 100px !important;
    font-size: 30px !important;
    margin: 0 auto !important;
    border-radius: 22px !important;
    display: flex;
    justify-content: center;
    align-items: center;
}
#main-desc {
    margin-top: 40px;
    padding: 32px 60px;
    background: #f8f8f8;
    border-radius: 20px;
    font-size: 20px;
    color: #333;
    text-align: center;
    box-shadow: 0 4px 24px #e0e7ef;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}

.gr-alert {
    background: #e0f2fe !important;
    color: #0f172a !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    text-align: center !important;
    margin: 14px auto !important;
    padding: 10px 18px !important;
    max-width: 320px;
    box-shadow: 0 2px 8px #38bdf844;
    transition: opacity 0.5s;
    letter-spacing: 1px;
}


"""

def image_to_base64(image):
    """将PIL图像转换为base64字符串"""
    if image is None:
        return None
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def base64_to_image(base64_str):
    """将base64字符串转换为PIL图像"""
    if not base64_str:
        return None
    
    # 移除data:image/...;base64,前缀
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

def api_tryon(
    human_image_base64: str,
    garment_image_base64: str,
    garment_description: str = "a shirt",
    auto_mask: bool = True,
    auto_crop: bool = False,
    denoise_steps: int = 25,
    seed: int = 42
) -> tuple[str, str]:
    """
    API版本的试穿函数
    
    参数:
        human_image_base64: 人物图片的base64编码字符串
        garment_image_base64: 服装图片的base64编码字符串
        garment_description: 服装描述
        auto_mask: 是否自动生成遮罩
        auto_crop: 是否自动裁剪
        denoise_steps: 去噪步骤数
        seed: 随机种子
    
    返回:
        tuple: (试穿结果图片base64, 遮罩图片base64)
    """
    try:
        # 转换base64为PIL图像
        human_img = base64_to_image(human_image_base64)
        garment_img = base64_to_image(garment_image_base64)
        
        if human_img is None or garment_img is None:
            raise ValueError("无效的图片数据")
        
        # 构造start_tryon函数需要的dict格式
        human_dict = {
            "background": human_img,
            "layers": [Image.new('RGB', human_img.size, (255, 255, 255))],  # 创建空白遮罩层
            "composite": None
        }
        
        # 调用原始的试穿函数
        result_img, mask_img = start_tryon(
            dict=human_dict,
            garm_img=garment_img,
            garment_des=garment_description,
            is_checked=auto_mask,
            is_checked_crop=auto_crop,
            denoise_steps=denoise_steps,
            seed=seed
        )
        
        # 转换结果为base64
        result_base64 = image_to_base64(result_img)
        mask_base64 = image_to_base64(mask_img)
        
        return result_base64, mask_base64
        
    except Exception as e:
        print(f"API试穿失败: {str(e)}")
        raise Exception(f"试穿处理失败: {str(e)}")

with gr.Blocks(css=custom_css) as demo:
    gr.HTML('<div id="global-bg-img"></div>')
    page_state = gr.State("home")
    fav_images = gr.State([])
    login_state = gr.State(False)  # 新增：全局登录状态

    # 首页
    with gr.Group(visible=True, elem_id="home-page") as home_group:
        gr.HTML('<div id="home-bg-img"></div>')
        fav_btn = gr.Button("收藏", elem_id="fav-btn")
        with gr.Row():
            register_btn = gr.Button("注册", elem_id="register-btn")
            login_btn = gr.Button("登录", elem_id="login-btn")
        with gr.Row(elem_id="center-btn-row"):
            mode1_btn = gr.Button("开始体验", elem_id="mode1-btn")
        gr.Markdown(
            """
            ### 欢迎使用 IDM-VTON 虚拟试衣系统
            简介：本系统支持上传人物和服装图片，体验AI驱动的虚拟试衣效果。

            ![展示图片](https://github.com/Shallow536/image/blob/main/new1.png?raw=true)
            """, 
            elem_id="main-desc")
        gr.Markdown(
            """
            ![展示图片](https://github.com/Shallow536/image/blob/main/new2.png?raw=true)""", 
            elem_id="main-desc")
        gr.Markdown(
            """
            **IDM-VTON是一种基于扩散模型的智能虚拟试衣技术，旨在通过人工智能生成高保真度的虚拟试穿效果。它结合了计算机视觉和深度学习技术，能够将服装图像自然地"穿"在目标人物身上，同时保持服装的纹理、褶皱和身体姿态的真实性。**""", 
            elem_id="main-desc")
        gr.Markdown(
            """
            ![展示图片](https://github.com/Shallow536/image/blob/main/new3.png?raw=true)""", 
            elem_id="main-desc")
        gr.Markdown(
            """
            #### 使用说明：
            1. 点击"开始体验"进入试衣页面。
            2. 在试衣页面，上传您的人物图片和服装图片。
            3. 调整参数，点击"试穿"查看效果。
            4. 可选：使用画笔工具手动标注遮罩区域。
            5. 点击"保存图片"下载结果。
            6. 返回首页可进行新一轮试穿或注册登录。
            """,
            elem_id="main-desc")

    # 登录页
    with gr.Group(visible=False, elem_id="login-page") as login_group:
        gr.Markdown("## 登录")
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                login_btn2 = gr.Button("登录", elem_id="login-btn2")
                username = gr.Textbox(label="账号", placeholder="请输入账号")
                password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                login_info = gr.Markdown("", elem_classes=["gr-alert"], visible=False)
                back_btn_login = gr.Button("返回首页", elem_id="back-btn-login")
            with gr.Column(scale=1):
                pass

    # 注册页
    with gr.Group(visible=False, elem_id="register-page") as register_group:
        gr.Markdown("## 注册新用户")
        nickname_reg = gr.Textbox(label="昵称", placeholder="请输入昵称")
        username_reg = gr.Textbox(label="账号", placeholder="请输入账号")
        password_reg = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
        password_reg2 = gr.Textbox(label="再次输入密码", type="password", placeholder="请再次输入密码")
        register_info = gr.Markdown("", elem_classes=["gr-alert"], visible=False)
        with gr.Row():
            register_confirm_btn = gr.Button("确定注册", elem_id="register-confirm-btn")
            back_btn_register = gr.Button("返回首页", elem_id="back-btn-register")

    # 模式一页（主功能页面）
    with gr.Group(visible=False, elem_id="mode1-page") as mode1_group:
        gr.Markdown("IDM-VTON 虚拟试穿")
        
        with gr.Row():
            with gr.Column():
                imgs = gr.ImageEditor(sources='upload', type="pil", label='人物遮罩：可手动涂抹或使用自动遮罩功能', interactive=True, height=650, width=540)
                # 人像图片文件夹快捷按钮区域
                with gr.Row():
                    browse_human_btn = gr.Button(" 浏览文件夹 ", elem_id="browse-human")
                    browse_human_file_btn = gr.Button(" 选择图片 ", elem_id="select-human")
                human_status = gr.Markdown("", elem_classes=["gr-alert"], visible=False)
                # 动态人像图片展示区域
                human_gallery = gr.Gallery(
                    label="人像图片浏览",
                    show_label=True,
                    elem_id="human_gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                    visible=False
                )
                
                with gr.Row():
                    is_checked = gr.Checkbox(label="自动生成遮罩", value=True)
                with gr.Row():
                    is_checked_crop = gr.Checkbox(label="自动裁剪", value=False)

            with gr.Column():
                garm_img = gr.Image(label="试穿服装", sources='upload', type="pil", height=650, width=540)
                # 服装图片文件夹快捷按钮区域
                with gr.Row():
                    browse_cloth_btn = gr.Button(" 浏览文件夹 ", elem_id="browse-cloth")
                    browse_cloth_file_btn = gr.Button(" 选择图片 ", elem_id="select-cloth")
                cloth_status = gr.Markdown("", elem_classes=["gr-alert"], visible=False)
                # 动态服装图片展示区域
                cloth_gallery = gr.Gallery(
                    label="服装图片浏览",
                    show_label=True,
                    elem_id="cloth_gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                    visible=False
                )
                
                with gr.Row(elem_id="prompt-container"):
                    prompt = gr.Textbox(placeholder="服装描述（例如：短袖圆领T恤）", label="服装描述", elem_id="prompt")

        with gr.Row():
            gr.Markdown("### 试穿参数设置")
        with gr.Row():
            with gr.Column(scale=1):
                denoise_steps = gr.Slider(label="去噪步骤", minimum=1, maximum=50, value=25, step=1)
            with gr.Column(scale=1):
                seed = gr.Number(label="随机种子", value=42, precision=0)
        with gr.Row():
            try_button = gr.Button("试穿", elem_id="try-btn")
            save_images_btn = gr.Button("保存图片", elem_id="save-img-btn")
        with gr.Row():
            with gr.Column():
                masked_img = gr.Image(label="遮罩图", type="pil", elem_id="mask-image", height=650, width=540)
            with gr.Column():
                image_out = gr.Image(label="试穿结果", type="pil", elem_id="result-image", height=650, width=540)
        
        # 图片保存区域
        with gr.Group():
            gr.Markdown("###  图片保存功能")
            with gr.Row():
                filename_prefix = gr.Textbox(
                    label="自定义文件名前缀", 
                    placeholder="输入文件名前缀，例如: my_tryon",
                    value="tryon_result",
                    scale=3
                )
            
            save_status = gr.Markdown("", elem_classes=["gr-alert"], visible=False)
        
        gr.Markdown('<div style="height: 20px;"></div>')
        with gr.Row():
            with gr.Column():
                pass
            with gr.Column(scale=2):
                fav_this_btn = gr.Button("收藏本次试穿", elem_id="fav-this-btn")
                fav_info = gr.Markdown("", elem_classes=["gr-alert", "fav-info-center"], visible=False)
            with gr.Column():
                pass
        back_btn1 = gr.Button("返回首页", elem_id="back-btn1")
        gr.Markdown("""
        #### 注意事项：
        - 请确保上传的图片清晰可见，服装图片尽量与人物姿势相符。
        - 自动生成遮罩可能不够精确，可手动调整。
        - 保存的图片为生成的试穿结果，可能与实际效果略有差异。
        """, elem_id="main-desc")

    # 收藏页
    with gr.Group(visible=False, elem_id="fav-page") as fav_group:
        gr.Markdown("## 我的收藏")
        fav_gallery = gr.Gallery(label="收藏的试穿图片", columns=3, height="600px", object_fit="cover", allow_preview=True)
        fav_del_info = gr.Markdown("", elem_classes=["gr-alert"], visible=False)
        del_btn = gr.Button("删除选中图片", elem_id="del-btn")
        gr.Markdown("点击图片后点击下方"'删除'"按钮即可删除")
        back_btn_fav = gr.Button("返回首页", elem_id="back-btn-fav")

    # 注册API端点
    gr.api(api_tryon, api_name="tryon")

    # 页面切换函数
    def switch_page(page):
        # 定义各页面初始状态
        clear_mode1_state = [
            None,  # imgs
            None,  # garm_img  
            "",    # prompt
            True,  # is_checked
            False, # is_checked_crop
            25,    # denoise_steps
            42,    # seed
            None,  # image_out
            None,  # masked_img
            "",    # human_status
            "",    # cloth_status
            gr.Gallery(visible=False),  # human_gallery
            gr.Gallery(visible=False),  # cloth_gallery
            "tryon_result",  # filename_prefix
            "",    # save_status
            "",    # fav_info
        ]
        
        clear_login_state = [
            "",    # username
            "",    # password
            "",    # login_info
        ]
        
        clear_register_state = [
            "",    # nickname
            "",    # username
            "",    # password
            "",    # password_reg2
            "",    # register_info
        ]
        
        clear_fav_state = [
            "",    # fav_del_info
            None,   # selected_idx，重置选中索引
            gr.update(selected=None), # fav_gallery取消选中
        ]
        
        if page == "mode1":
            return (
                gr.update(visible=page=="home"),
                gr.update(visible=page=="login"),
                gr.update(visible=page=="mode1"),
                gr.update(visible=page=="fav"),
                gr.update(visible=page=="register"),
                *clear_mode1_state
            )
        elif page == "login":
            return (
                gr.update(visible=page=="home"),
                gr.update(visible=page=="login"),
                gr.update(visible=page=="mode1"),
                gr.update(visible=page=="fav"),
                gr.update(visible=page=="register"),
                *clear_login_state
            )
        elif page == "register":
            return (
                gr.update(visible=page=="home"),
                gr.update(visible=page=="login"),
                gr.update(visible=page=="mode1"),
                gr.update(visible=page=="fav"),
                gr.update(visible=page=="register"),
                *clear_register_state
            )
        elif page == "fav":
            # 加载收藏图片
            collection_files, _ = load_collection_images()
            return (
                gr.update(visible=page=="home"),
                gr.update(visible=page=="login"),
                gr.update(visible=page=="mode1"),
                gr.update(visible=page=="fav"),
                gr.update(visible=page=="register"),
                gr.update(value=collection_files),  # 更新收藏画廊
                *clear_fav_state[1:]  # 跳过第一个元素，因为已经更新了fav_gallery
            )
        else:  # home
            return (
                gr.update(visible=page=="home"),
                gr.update(visible=page=="login"),
                gr.update(visible=page=="mode1"),
                gr.update(visible=page=="fav"),
                gr.update(visible=page=="register"),
            )

    # ================== 事件绑定 ==================
    # 试穿按钮
    try_button.click(
        fn=start_tryon,
        inputs=[imgs, garm_img, prompt, is_checked, is_checked_crop, denoise_steps, seed],
        outputs=[image_out, masked_img],
        api_name='tryon'
    )
    
    # 图片保存
    save_images_btn.click(fn=save_generated_image, inputs=[image_out, masked_img, filename_prefix], outputs=save_status)
    
    # 人像图片相关
    browse_human_btn.click(fn=lambda: update_human_gallery(), outputs=[human_gallery, human_status])
    browse_human_file_btn.click(fn=select_human_image, outputs=[imgs, human_status])
    human_gallery.select(fn=select_from_human_gallery, outputs=[imgs, human_status])
    
    # 服装图片相关
    browse_cloth_btn.click(fn=lambda: update_cloth_gallery(), outputs=[cloth_gallery, cloth_status])
    browse_cloth_file_btn.click(fn=select_garment_image, outputs=[garm_img, cloth_status])
    cloth_gallery.select(fn=select_from_cloth_gallery, outputs=[garm_img, cloth_status])

    # 收藏页事件绑定
    selected_idx = gr.State(None)
    def save_selected(evt: gr.SelectData):
        return evt.index
    fav_gallery.select(save_selected, None, selected_idx)
    
    def delete_by_btn(idx):
        """删除选中的收藏图片文件"""
        try:
            current_files, _ = load_collection_images()
            if current_files and idx is not None and 0 <= idx < len(current_files):
                file_to_delete = current_files[idx]
                os.remove(file_to_delete)
                # 重新加载图片列表
                updated_files, status = load_collection_images()
                return gr.update(value="已删除！", visible=True), None, gr.update(value=updated_files)
            return gr.update(value="请选择要删除的图片", visible=True), idx, gr.update()
        except Exception as e:
            return gr.update(value=f"删除失败: {str(e)}", visible=True), idx, gr.update()
    
    del_btn.click(delete_by_btn, [selected_idx], [fav_del_info, selected_idx, fav_gallery])
    def hide_fav_del_info():
        time.sleep(2)
        return gr.update(value="", visible=False)
    del_btn.click(hide_fav_del_info, None, fav_del_info, show_progress=False, queue=True)
    
    # 收藏本次试穿 - 修改为保存到文件
    def add_to_collection_with_save(img):
        """收藏图片并保存到文件系统"""
        if img is not None:
            status, file_path = save_to_collection(img, "collection")
            # 重新加载收藏画廊
            updated_files, _ = load_collection_images()
            return gr.update(value=status, visible=True), gr.update(value=updated_files)
        else:
            return gr.update(value="❌ 没有可收藏的图片", visible=True), gr.update()
    
    fav_this_btn.click(add_to_collection_with_save, [image_out], [fav_info, fav_gallery])
    def hide_fav_info():
        time.sleep(2)
        return gr.update(value="", visible=False)
    fav_this_btn.click(hide_fav_info, None, fav_info, show_progress=False, queue=True)

    # 登录页按钮事件
    # 1. login_action 返回登录结果
    def login_action(user, pwd):
        if not user or not pwd:
            return gr.update(value="请输入账号和密码", visible=True), False, gr.update(value="登录")
        success = external_login_api(user, pwd)
        if success:
            return gr.update(value="登录成功！", visible=True), True, gr.update(value="退出登录")
        else:
            return gr.update(value="登录失败，账号或密码错误", visible=True), False, gr.update(value="登录")

    login_btn2.click(
        login_action,
        inputs=[username, password],
        outputs=[login_info, login_state, login_btn]
    )

    # 2. login_success_jump 只在登录成功时跳转主页
    def login_success_jump(state):
        time.sleep(2)
        if state:
            # 登录成功，2秒后跳主页
            return gr.update(value="", visible=False), *switch_page("home")
        else:
            # 登录失败，只清空提示，不跳转
            return gr.update(value="", visible=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    login_btn2.click(
        login_success_jump,
        inputs=[login_state],
        outputs=[login_info, home_group, login_group, mode1_group, fav_group, register_group],
        show_progress=False, queue=True
    )

    # 注册页按钮事件
    def register_action(nickname, user, pwd1, pwd2):
        if not nickname or not user or not pwd1 or not pwd2:
            return gr.update(value="请填写所有信息", visible=True), *switch_page("register")
        if pwd1 != pwd2:
            return gr.update(value="两次输入的密码不一致", visible=True), *switch_page("register")
        success = external_register_api(user, pwd1, nickname)
        if success:
            # 注册成功，先显示提示，不切换页面
            return gr.update(value="注册成功！", visible=True), *switch_page("register")
        else:
            return gr.update(value="注册失败，用户名已存在或其他原因", visible=True), *switch_page("register")
    register_confirm_btn.click(
        register_action,
        inputs=[nickname_reg, username_reg, password_reg, password_reg2],
        outputs=[register_info, home_group, login_group, mode1_group, fav_group, register_group]
    )
    def register_success_jump():
        time.sleep(2)
        # 2秒后跳转主页并清空提示
        return gr.update(value="", visible=False), *switch_page("home")
    register_confirm_btn.click(register_success_jump, None, [register_info, home_group, login_group, mode1_group, fav_group, register_group], show_progress=False, queue=True)

    # 主页登录/退出登录按钮切换逻辑
    def home_login_btn_action(btn_value, state):
        if btn_value == "退出登录" and is_logged_in(state):
            # 退出登录，状态变为False，按钮变回“登录”
            return False, gr.update(value="登录")
        elif btn_value == "登录":
            # 跳转到登录页，状态不变
            return state, gr.update()
        else:
            return state, gr.update()
    login_btn.click(
        home_login_btn_action,
        inputs=[login_btn, login_state],
        outputs=[login_state, login_btn]
    )
    # 跳转到登录页逻辑
    def jump_to_login(btn_value, state):
        if btn_value == "登录" and not is_logged_in(state):
            return switch_page("login")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    login_btn.click(
        jump_to_login,
        inputs=[login_btn, login_state],
        outputs=[home_group, login_group, mode1_group, fav_group, register_group]
    )

    # ================== 页面切换按钮事件绑定 ==================
    # 首页按钮
    mode1_btn.click(
        lambda: switch_page("mode1"), 
        None, 
        [home_group, login_group, mode1_group, fav_group, register_group,
         imgs, garm_img, prompt, is_checked, is_checked_crop, denoise_steps, seed,
         image_out, masked_img, human_status, cloth_status, human_gallery, cloth_gallery,
         filename_prefix, save_status, fav_info]
    )
    fav_btn.click(
        lambda: switch_page("fav"), 
        None, 
        [home_group, login_group, mode1_group, fav_group, register_group, 
         fav_gallery, fav_del_info]  # 修改输出，包含fav_gallery更新
    )
    login_btn.click(
        lambda: switch_page("login"), 
        None, 
        [home_group, login_group, mode1_group, fav_group, register_group,
         username, password, login_info]
    )
    register_btn.click(
        lambda: switch_page("register"), 
        None, 
        [home_group, login_group, mode1_group, fav_group, register_group,
         nickname_reg, username_reg, password_reg, password_reg2, register_info]
    )
    # 返回首页按钮
    back_btn1.click(
        lambda: switch_page("home"), 
        None, 
        [home_group, login_group, mode1_group, fav_group, register_group]
    )
    back_btn_login.click(
        lambda: switch_page("home"), 
        None, 
        [home_group, login_group, mode1_group, fav_group, register_group]
    )
    back_btn_register.click(
        lambda: switch_page("home"), 
        None, 
        [home_group, login_group, mode1_group, fav_group, register_group]
    )
    def reset_fav_gallery():
        return gr.update(selected=None)
    # 先取消选中，再跳转首页
    back_btn_fav.click(reset_fav_gallery, None, fav_gallery)
    back_btn_fav.click(
        lambda: switch_page("home"), 
        None, 
        [home_group, login_group, mode1_group, fav_group, register_group]
    )



if __name__ == "__main__":
    demo.launch(show_api=True, share=False, allowed_paths=allowed_paths)