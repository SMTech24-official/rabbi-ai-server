from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from rembg import remove, new_session
import io
from typing import Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

app = FastAPI(title="ID Photo Background Processor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://143.198.214.226:3000",
                    "https://autoidgen.com",
                    "http://localhost:3001",
                    "http://localhost:3000",
                    "https://autoidgen.com"
                    "https://www.autoidgen.com"
                     , "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rembg session for human segmentation (best for ID photos)
bg_removal_session = new_session('u2net_human_seg')

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

class IDPhotoProcessor:
    @staticmethod
    def enhance_image_quality(image: Image.Image) -> Image.Image:
        """Enhance image quality for ID photos"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        return image
    
    @staticmethod
    def resize_for_id_photo(image: Image.Image, size: Tuple[int, int] = (600, 800)) -> Image.Image:
        """Resize image maintaining aspect ratio and quality for ID photos"""
        return image.resize(size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def remove_background_high_quality(image: Image.Image) -> Image.Image:
        """Remove background with high quality settings"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr = img_byte_arr.getvalue()
        result = remove(img_byte_arr, session=bg_removal_session)
        return Image.open(io.BytesIO(result)).convert('RGBA')
    
    @staticmethod
    def add_background_color(image: Image.Image, color: str) -> Image.Image:
        """Add solid background color maintaining quality"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[-1])
        return background
    
    @staticmethod
    def smooth_edges(image: Image.Image) -> Image.Image:
        """Apply subtle edge smoothing"""
        if image.mode == 'RGBA':
            img_array = np.array(image)
            alpha_channel = img_array[:, :, 3]
            smoothed_alpha = cv2.GaussianBlur(alpha_channel, (3, 3), 0.5)
            img_array[:, :, 3] = smoothed_alpha
            return Image.fromarray(img_array, 'RGBA')
        return image
    
    @staticmethod
    def center_face_in_frame(image: Image.Image) -> Image.Image:
        """Detect face and center it in the frame"""
        try:
            img_array = np.array(image)
            if image.mode == 'RGBA':
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                img_h, img_w = img_array.shape[:2]
                target_y = img_h // 3
                target_x = img_w // 2
                shift_x = target_x - face_center_x
                shift_y = target_y - face_center_y
                if abs(shift_x) > 20 or abs(shift_y) > 20:
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    if image.mode == 'RGBA':
                        shifted = cv2.warpAffine(img_array, M, (img_w, img_h), borderMode=cv2.BORDER_TRANSPARENT)
                        return Image.fromarray(shifted, 'RGBA')
                    else:
                        shifted = cv2.warpAffine(img_cv, M, (img_w, img_h), borderValue=(255, 255, 255))
                        shifted_rgb = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(shifted_rgb, 'RGB')
            return image
        except Exception as e:
            logging.warning(f"Face centering failed: {e}")
            return image

processor = IDPhotoProcessor()

@app.post("/process-id-photo")
async def process_id_photo(
    file: UploadFile = File(...),
    background_color: str = Form("#FFFFFF"),
    width: int = Form(600),
    height: int = Form(800),
    enhance_quality: bool = Form(True),
    center_face: bool = Form(True)
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        def process_image():
            current_image = image.copy()
            if enhance_quality:
                current_image = processor.enhance_image_quality(current_image)
            current_image = processor.remove_background_high_quality(current_image)
            current_image = processor.smooth_edges(current_image)
            if center_face:
                current_image = processor.center_face_in_frame(current_image)
            current_image = processor.resize_for_id_photo(current_image, (width, height))
            final_image = processor.add_background_color(current_image, background_color)
            return final_image
        final_image = await asyncio.get_event_loop().run_in_executor(executor, process_image)
        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format='PNG', quality=95, optimize=False, dpi=(300, 300))
        img_byte_arr.seek(0)
        return StreamingResponse(
            io.BytesIO(img_byte_arr.read()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=id_photo.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/remove-background-only")
async def remove_background_only(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        def process_image():
            enhanced = processor.enhance_image_quality(image)
            result = processor.remove_background_high_quality(enhanced)
            return processor.smooth_edges(result)
        result = await asyncio.get_event_loop().run_in_executor(executor, process_image)
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG', quality=95, dpi=(300, 300))
        img_byte_arr.seek(0)
        return StreamingResponse(
            io.BytesIO(img_byte_arr.read()),
            media_type="image/png"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "ID Photo Background Processor API",
        "endpoints": {
            "/process-id-photo": "Complete ID photo processing",
            "/remove-background-only": "Background removal only",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)