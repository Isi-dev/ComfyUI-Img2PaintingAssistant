import cv2
import numpy as np
import torch



class Painting():

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "painting_details": ("FLOAT", {"default": 26, "min": 1, "max": 36, "step": 1}),
                "painting_blur": ("INT", {"default": 1, "min": 1, "max": 9, "step": 2}),
                "sharpness": ("INT", {"default": 7, "min": 1, "max": 11, "step": 2}),
                "brightness": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.1}),
                "hue": ("FLOAT", {"default": 0, "min": 0, "max": 179, "step": 1}),
                "saturation": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 10, "step": 0.1}),
                "lightness": ("FLOAT", {"default": 1.4, "min": 0.1, "max": 10, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "correct_black_Img": ("BOOLEAN", { "default": False }),
                          
            },
            "optional": {              
                "lineArt": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("painting", "sharpImage")
    FUNCTION = "process"
    CATEGORY = "image"

    def process(self, image, painting_details, painting_blur, sharpness, brightness, hue, saturation, lightness, contrast, correct_black_Img, lineArt=None):
        imgNo =  image.shape[0]
        if imgNo == 1:
            print("1 image received for conversion to Painting")
        else:
            print(f"{imgNo} images received for conversion to Painting")
        paintings = []
        paintings2 = []

        no_lineArt = True

        if lineArt is not None:
            no_lineArt = False

        if correct_black_Img:
            image = image*255

        if no_lineArt:
            for img in image:
                painting, painting2 = processImg2Painting(img, painting_details, painting_blur, sharpness, brightness, hue, saturation, lightness, contrast, lineArt)
                paintings.append(painting)
                paintings2.append(painting2)
        else:
            for img, lineart in zip(image, lineArt):
                painting, painting2 = processImg2Painting(img, painting_details, painting_blur, sharpness, brightness, hue, saturation, lightness, contrast, lineart)
                paintings.append(painting)
                paintings2.append(painting2)
        
        paintings = torch.cat(paintings, dim=0)
        paintings2 = torch.cat(paintings2, dim=0)
        
        print("Conversion complete!")

        return (paintings, paintings2)   
    

class ProcessInspyrenetRembg:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "use_bg_color": ("BOOLEAN", { "default": False }),
                "bg_color": (["white", "black", "red", "lime", "blue", "yellow", "cyan", "magenta", "silver", "gray", "maroon", "olive", "green", "purple", "teal", "navy"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, mask, use_bg_color, bg_color):
        
        processedImg = create_transparent_images(image, mask, use_bg_color, bg_color)

        return (processedImg, mask)


def processImg2Painting(imageM, details, blur, sharpness, lightness, hue, saturation, light, contrast, lineArt):
    if imageM is None:
        raise ValueError("Input image is required")

    if isinstance(imageM, torch.Tensor):
        imageM = imageM.squeeze(0).cpu().numpy()
        imageM = convert_to_uint8(imageM)

    if imageM.ndim not in [2, 3]:
        raise ValueError("Input image must be 2D or 3D numpy array")
    
    
    has_alpha = imageM.shape[-1] == 4
    if has_alpha:
        alpha_channel = imageM[:, :, 3]  # Extract the alpha channel
        image = imageM[:, :, :3]  # Keep only the RGB channels
    else:
        image = imageM


    

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)

    h = (h + hue) % 180

    s = np.clip(s * saturation, 0, 255)

    v = np.clip(v * lightness, 0, 255).astype(np.uint8)

    if h.dtype != np.uint8:
        h = h.astype(np.uint8)
    if s.dtype != np.uint8:
        s = s.astype(np.uint8)
    if v.dtype != np.uint8:
        v = v.astype(np.uint8)

    modified_hsv = cv2.merge([h, s, v])

    image = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

    details = 51 - details
    details = details/10

    imageP = cv2.resize(image, (int(image.shape[1] * 1 / details), int(image.shape[0] * 1 / details)), interpolation=cv2.INTER_AREA)

    cleaner_image = cv2.medianBlur(imageP, blur)
    for i in range(2):
        cleaner_image = cv2.medianBlur(cleaner_image, blur)

    filtered_image = cv2.bilateralFilter(cleaner_image, 3, 10, 5)

    for i in range(2):
        filtered_image = cv2.bilateralFilter(filtered_image, 3, 20, 10)

    for i in range(3):
        filtered_image = cv2.bilateralFilter(filtered_image, 5, 30, 10)

    gaussian_mask = cv2.GaussianBlur(filtered_image, (sharpness, sharpness), 2)
    sharper_image = cv2.addWeighted(filtered_image, light, gaussian_mask, -0.5, 0)
    sharper_image = cv2.addWeighted(sharper_image, 1.4, gaussian_mask, -0.2, 10)

    gaussian_maskP = cv2.GaussianBlur(sharper_image, (sharpness, sharpness), 2)
    sharper_imageP = cv2.addWeighted(sharper_image, light, gaussian_maskP, -0.5, 0)
    sharper_imageP = cv2.addWeighted(sharper_imageP, 1.4, gaussian_maskP, -0.2, 10)

    gaussian_maskPP = cv2.GaussianBlur(sharper_imageP, (sharpness, sharpness), 2)
    sharper_imagePP = cv2.addWeighted(sharper_imageP, light, gaussian_maskPP, -0.5, 0)
    sharper_imagePP = cv2.addWeighted(sharper_imagePP, 1.4, gaussian_maskPP, -0.2, 10)


    if sharpness == 9:
        sharper_image = sharper_imageP
    if sharpness > 9:
        sharper_image = sharper_imagePP


    sharper_image = cv2.resize(sharper_image, (int(sharper_image.shape[1] * details), int(sharper_image.shape[0] * details)))

    sharper_image = contrast_image(sharper_image, contrast)


    if lineArt is not None:
        lineArt = lineArt.squeeze(0).cpu().numpy()
        lineArt = convert_to_uint8(lineArt)
        sharper_image = cv2.resize(sharper_image, (int(lineArt.shape[1]), int(lineArt.shape[0])))
        mask = (lineArt[..., :3] < 240).any(axis=-1)
        sharper_image[mask] = lineArt[mask]


    gaussian_mask2 = cv2.GaussianBlur(image, (sharpness, sharpness), 2)
    sharper_image2 = cv2.addWeighted(image, light, gaussian_mask2, -0.5, 0)
    sharper_image2 = cv2.addWeighted(sharper_image2, 1.4, gaussian_mask2, -0.2, 10)

    gaussian_mask3 = cv2.GaussianBlur(sharper_image2, (sharpness, sharpness), 2)
    sharper_image3 = cv2.addWeighted(sharper_image2, light, gaussian_mask3, -0.5, 0)
    sharper_image3 = cv2.addWeighted(sharper_image3, 1.4, gaussian_mask3, -0.2, 10)

    gaussian_mask4 = cv2.GaussianBlur(sharper_image3, (sharpness, sharpness), 2)
    sharper_image4 = cv2.addWeighted(sharper_image3, light, gaussian_mask4, -0.5, 0)
    sharper_image4 = cv2.addWeighted(sharper_image4, 1.4, gaussian_mask4, -0.2, 10)

    if has_alpha:
        sharper_image = cv2.resize(sharper_image, (int(sharper_image2.shape[1]), int(sharper_image2.shape[0])))

    sharper_image = torch.from_numpy(sharper_image).permute(2, 0, 1).unsqueeze(0).float()/255
    sharper_image = sharper_image.permute(0, 2, 3, 1)

    if sharpness == 9:
        sharper_image2 = sharper_image3
    if sharpness > 9:
        sharper_image2 = sharper_image4
    
    sharper_image2 = contrast_image(sharper_image2, contrast)

    sharper_image2 = torch.from_numpy(sharper_image2).permute(2, 0, 1).unsqueeze(0).float()/255
    sharper_image2 = sharper_image2.permute(0, 2, 3, 1)

    if has_alpha:
        alpha_channel = alpha_channel / 255.0  
        alpha_channel = np.expand_dims(alpha_channel, axis=-1)  
        alpha_channel = np.expand_dims(alpha_channel, axis=0)  

        # Concatenate the alpha channel to the processed image
        sharper_image = np.concatenate([sharper_image.numpy(), alpha_channel], axis=-1)
        sharper_image2 = np.concatenate([sharper_image2.numpy(), alpha_channel], axis=-1)

        sharper_image = torch.from_numpy(sharper_image).float()
        sharper_image2 = torch.from_numpy(sharper_image2).float()

    return sharper_image, sharper_image2


def create_transparent_images(img_stack, mask, use_bg_color, bg_color):
    """
    Create a batch of images with transparent backgrounds using img_stack and mask.

    Args:
    - img_stack (torch.Tensor): A batch of images with shape (B, H, W, 4) (RGBA).
    - mask (torch.Tensor): A batch of masks with shape (B, H, W), where non-zero values indicate the foreground.

    Returns:
    - torch.Tensor: A batch of images with shape (B, H, W, 4) (RGBA), with the background set to transparent.
    """
   
    if img_stack.shape[-1] != 4:
        raise ValueError("The input img_stack must have 4 channels (RGBA).")
    
    if mask.ndim != 3:
        raise ValueError("The mask must have 3 dimensions (B, H, W).")
    
    binary_mask = (mask > 0.5).float()

    rgb_stack = img_stack[:, :, :, :3]  # Shape: (B, H, W, 3)

    if use_bg_color is False:
        rgb_stack = rgb_stack * mask.unsqueeze(-1)
    else:
        bg_color = getColor(bg_color)
        bg_color = torch.tensor(bg_color).float().reshape(1, 1, 1, 3)  # Shape: (1, 1, 1, 3)

        rgb_stack = rgb_stack * binary_mask.unsqueeze(-1) + bg_color * (1 - binary_mask).unsqueeze(-1)

    new_alpha_channel = binary_mask.unsqueeze(-1) # Shape: (B, H, W, 1)
    
    rgba_stack = torch.cat([rgb_stack, new_alpha_channel], dim=-1)  # Shape: (B, H, W, 4)
    
    return rgba_stack


def convert_to_uint8(image):  
    if image.dtype == np.uint8:
        return image
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def contrast_image(data, scale):
    image = data.astype(np.float32)
    contrast_image = scale * (image - 128) + 128
    contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)

    return contrast_image

def getColor(col):
    color_dict = {
        "white": [255, 255, 255],
        "black": [0, 0, 0],
        "red": [255, 0, 0],
        "lime": [0, 255, 0],
        "blue": [0, 0, 255],
        "yellow": [255, 255, 0],
        "cyan": [0, 255, 255],
        "magenta": [255, 0, 255],
        "silver": [192, 192, 192],
        "gray": [128, 128, 128],
        "maroon": [128, 0, 0],
        "olive": [128, 128, 0],
        "green": [0, 128, 0],
        "purple": [128, 0, 128],
        "teal": [0, 128, 128],
        "navy": [0, 0, 128],
    }

    return color_dict.get(col, [0, 0, 0])
    

NODE_CLASS_MAPPINGS = {
    "Painting" : Painting,
    "ProcessInspyrenetRembg" : ProcessInspyrenetRembg,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Painting" :"Image Painting Assitant",
    "ProcessInspyrenetRembg" : "Inspyrenet Rembg Assistant",
    
}
