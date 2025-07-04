# IDM-VTON 虚拟试穿 API 文档

## 概述

IDM-VTON API 提供了强大的虚拟试穿功能，允许用户通过HTTP请求将服装图片"穿"到人物图片上，生成高质量的试穿效果图。

## API 端点

### 基础信息
- **服务地址**: `http://localhost:7860` (默认端口)
- **API 名称**: `/tryon`
- **请求方法**: POST
- **内容类型**: application/json

## 接口参数

### api_tryon 函数参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `human_image_base64` | string | ✅ | - | 人物图片的base64编码字符串 |
| `garment_image_base64` | string | ✅ | - | 服装图片的base64编码字符串 |
| `garment_description` | string | ❌ | "a shirt" | 服装描述文本，用于引导生成 |
| `auto_mask` | boolean | ❌ | true | 是否自动生成遮罩 |
| `auto_crop` | boolean | ❌ | false | 是否自动裁剪图片 |
| `denoise_steps` | integer | ❌ | 25 | 去噪步骤数，范围1-50 |
| `seed` | integer | ❌ | 42 | 随机种子，用于结果可重现 |

### 参数详细说明

#### human_image_base64
- **格式**: `data:image/[jpeg|png];base64,[base64数据]`
- **要求**: 
  - 图片应包含完整的人物形象
  - 人物姿态清晰，便于姿态检测
  - 推荐分辨率: 512x768 或更高
  - 支持格式: JPEG, PNG

#### garment_image_base64
- **格式**: `data:image/[jpeg|png];base64,[base64数据]`
- **要求**:
  - 服装图片应背景干净
  - 服装完整展示
  - 推荐分辨率: 512x768 或更高
  - 支持格式: JPEG, PNG

#### garment_description
- **示例**: 
  - `"a white shirt"`
  - `"a blue dress"`
  - `"a red jacket"`
  - `"短袖圆领T恤"`
- **作用**: 帮助AI理解服装类型，提高生成质量

#### auto_mask
- **true**: 使用AI自动检测人物身体部位并生成遮罩
- **false**: 使用手动绘制的遮罩（需要在human_image中提供）
- **注意**: 如果姿态检测失败，系统会自动切换到手动模式

#### auto_crop
- **true**: 自动裁剪图片以适应标准比例
- **false**: 保持原始图片比例

#### denoise_steps
- **范围**: 1-50
- **建议值**: 
  - 快速预览: 15-20
  - 标准质量: 25-30
  - 高质量: 35-50
- **注意**: 步骤越多，生成时间越长但质量越高

#### seed
- **范围**: 任意整数
- **作用**: 确保相同参数下生成相同结果
- **随机生成**: 可使用时间戳或随机数

## 返回值

### 成功响应
```json
{
  "data": [
    "data:image/png;base64,[试穿结果图片base64]",
    "data:image/png;base64,[遮罩图片base64]"
  ]
}
```

### 返回值说明
- **第一个元素**: 试穿结果图片（base64编码）
- **第二个元素**: 生成的遮罩图片（base64编码）

### 错误响应
```json
{
  "error": "错误描述信息"
}
```

## 使用示例

### 1. 使用 gradio_client (推荐)

```python
from gradio_client import Client
import base64

# 连接到服务
client = Client("http://localhost:7860")

# 图片转base64函数
def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"

# 准备图片
human_base64 = image_to_base64("path/to/human.jpg")
garment_base64 = image_to_base64("path/to/garment.jpg")

# 调用API
result = client.predict(
    human_image_base64=human_base64,
    garment_image_base64=garment_base64,
    garment_description="a white shirt",
    auto_mask=True,
    auto_crop=False,
    denoise_steps=25,
    seed=42,
    api_name="/tryon"
)

# 处理结果
result_image_base64, mask_image_base64 = result
```

### 2. 使用 requests

```python
import requests
import base64

def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"

# 准备数据
human_base64 = image_to_base64("path/to/human.jpg")
garment_base64 = image_to_base64("path/to/garment.jpg")

# 构建请求
payload = {
    "data": [
        human_base64,           # human_image_base64
        garment_base64,         # garment_image_base64
        "a white shirt",        # garment_description
        True,                   # auto_mask
        False,                  # auto_crop
        25,                     # denoise_steps
        42                      # seed
    ]
}

# 发送请求
response = requests.post(
    "http://localhost:7860/api/tryon",
    json=payload,
    timeout=300
)

# 处理响应
if response.status_code == 200:
    result = response.json()
    result_image_base64 = result['data'][0]
    mask_image_base64 = result['data'][1]
else:
    print(f"请求失败: {response.status_code}")
```

### 3. JavaScript 示例

```javascript
// 图片转base64
function imageToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// 调用API
async function tryonAPI(humanFile, garmentFile) {
    const humanBase64 = await imageToBase64(humanFile);
    const garmentBase64 = await imageToBase64(garmentFile);
    
    const payload = {
        data: [
            humanBase64,
            garmentBase64,
            "a white shirt",
            true,    // auto_mask
            false,   // auto_crop
            25,      // denoise_steps
            42       // seed
        ]
    };
    
    const response = await fetch('http://localhost:7860/api/tryon', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    });
    
    const result = await response.json();
    return result.data; // [resultImageBase64, maskImageBase64]
}
```

## 最佳实践


### 参数优化建议

1. **快速测试**: 
   ```python
   denoise_steps=15, auto_mask=False
   ```

2. **标准质量**:
   ```python
   denoise_steps=25, auto_mask=True
   ```

3. **高质量输出**:
   ```python
   denoise_steps=40, auto_mask=True
   ```

### 错误处理

```python
try:
    result = client.predict(...)
    if result and len(result) >= 2:
        result_image, mask_image = result
        # 处理成功结果
    else:
        print("API返回格式错误")
except Exception as e:
    if "list index out of range" in str(e):
        print("姿态检测失败，请尝试更清晰的人物图片")
    elif "CUDA" in str(e):
        print("GPU内存不足，请稍后重试")
    else:
        print(f"处理失败: {str(e)}")
```

