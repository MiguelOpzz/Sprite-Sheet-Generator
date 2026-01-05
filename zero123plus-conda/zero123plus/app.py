import os
import sys
import numpy
import torch
import rembg
import threading
import urllib.request
from PIL import Image
import streamlit as st
import json
import math
from io import BytesIO
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Increase recursion limit to prevent recursion errors
sys.setrecursionlimit(10000)

img_example_counter = 0
iret_base = 'resources/examples'
# Make sure the examples directory exists
if os.path.exists(iret_base):
    iret = [
        dict(rimageinput=os.path.join(iret_base, x), dispi=os.path.join(iret_base, x))
        for x in sorted(os.listdir(iret_base))
    ]
else:
    iret = []
    st.warning(f"Examples directory '{iret_base}' not found. Please create it and add example images.")


class SAMAPI:
    predictor = None

    @staticmethod
    @st.cache_resource
    def get_instance(sam_checkpoint=None):
        if SAMAPI.predictor is None:
            if sam_checkpoint is None:
                sam_checkpoint = "tmp/sam_vit_h_4b8939.pth"
            if not os.path.exists(sam_checkpoint):
                os.makedirs('tmp', exist_ok=True)
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    sam_checkpoint
                )
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_type = "default"

            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            SAMAPI.predictor = predictor
        return SAMAPI.predictor

    @staticmethod
    def segment_api(rgb, mask=None, bbox=None, sam_checkpoint=None):
        np = numpy
        predictor = SAMAPI.get_instance(sam_checkpoint)
        predictor.set_image(rgb)
        if mask is None and bbox is None:
            box_input = None
        else:
            # mask to bbox
            if bbox is None:
                y1, y2, x1, x2 = np.nonzero(mask)[0].min(), np.nonzero(mask)[0].max(), np.nonzero(mask)[1].min(), \
                                 np.nonzero(mask)[1].max()
            else:
                x1, y1, x2, y2 = bbox
            box_input = np.array([[x1, y1, x2, y2]])
        masks, scores, logits = predictor.predict(
            box=box_input,
            multimask_output=True,
            return_logits=False,
        )
        mask = masks[-1]
        return mask

def segment_img(img: Image):
    output = rembg.remove(img)
    mask = numpy.array(output)[:, :, 3] > 0
    sam_mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
    segmented_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    segmented_img.paste(img, mask=Image.fromarray(sam_mask))
    return segmented_img


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def separate_orthographic_views(zero123pp_imgs):
    """Separate the generated viewpoints into orthographic-like views for game assets"""

    imgs = [
        zero123pp_imgs.crop([0, 0, 320, 320]),      # Front view
        zero123pp_imgs.crop([0, 640, 320, 960]),    # Right side view
        zero123pp_imgs.crop([320, 320, 640, 640]),    # Back view
        zero123pp_imgs.crop([320, 0, 640, 320]),  # Left side view
        zero123pp_imgs.crop([0, 320, 320, 640]),    # Top view
        zero123pp_imgs.crop([320, 640, 640, 960])   # Bottom view
    ]
    return imgs


def create_custom_view_layout(images, selected_views):
    """Create a custom layout based on selected views"""
    if not selected_views:
        return None
      
    view_info = {
        "front": {"image": images[0], "name": "Front Orthographic", "description": "Primary front view"},
        "right": {"image": images[1], "name": "Right Orthographic", "description": "Right side elevation"},
        "back": {"image": images[2], "name": "Back Orthographic", "description": "Back elevation view"},
        "left": {"image": images[3], "name": "Left Orthographic", "description": "Left side elevation"},
        "top": {"image": images[4], "name": "Top Orthographic", "description": "Top plan view"},
        "bottom": {"image": images[5], "name": "Bottom Orthographic", "description": "Bottom view"}
    }
    
    # Filter only selected views
    custom_views = {}
    for view_name in selected_views:
        if view_name in view_info:
            custom_views[view_name] = view_info[view_name]
    
    return custom_views


def remove_background_single(img):
    """Remove background from a single image and ensure proper RGBA format"""
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Use rembg to remove background
        output = rembg.remove(img)
        
        # Ensure output is RGBA
        if output.mode != 'RGBA':
            output = output.convert('RGBA')
        
        return output
            
    except Exception as e:
        st.warning(f"Background removal warning: {e}")
        # Fallback: convert to RGBA with full opacity
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img


def ensure_proper_transparency(img):
    """Ensure image has proper transparency mask"""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Extract alpha channel
    alpha = img.split()[-1]
    
    # Convert to numpy array for processing
    alpha_array = numpy.array(alpha)
    
    # Ensure we have a proper binary mask (0 for transparent, 255 for opaque)
    mask = Image.fromarray(alpha_array)
    
    return img, mask


def create_sprite_sheet(views_dict, sprite_layout="horizontal", frame_size=256, spacing=10):
    """
    Create sprite sheets from orthographic views with proper transparency handling
    
    Args:
        views_dict: Dictionary of views with keys like "front", "top", "right", etc.
        sprite_layout: "horizontal", "vertical", "grid", or "2x3_grid"
        frame_size: Size of each frame in the sprite sheet
        spacing: Spacing between frames
    """
    # Define the order for all 6 views
    all_view_order = ["front", "right", "back", "left", "top", "bottom"]
    
    # Filter and order views for sprite sheet
    sprite_views = []
    for view_name in all_view_order:
        if view_name in views_dict:
            sprite_views.append(views_dict[view_name]["image"])
    
    if not sprite_views:
        return None
    
    # Process all views for sprite sheet
    processed_views = []
    for view in sprite_views:
        try:
            # Remove background and ensure proper transparency
            view_nobg = remove_background_single(view)
            
            # Resize to frame size
            view_resized = view_nobg.resize((frame_size, frame_size), Image.Resampling.LANCZOS)
            
            # Ensure proper transparency
            view_final, mask = ensure_proper_transparency(view_resized)
            processed_views.append((view_final, mask))
            
        except Exception as e:
            st.warning(f"Error processing view for sprite sheet: {e}")
            # Fallback: create a blank transparent frame
            blank_frame = Image.new("RGBA", (frame_size, frame_size), (0, 0, 0, 0))
            processed_views.append((blank_frame, blank_frame.split()[-1]))
    
    # Create sprite sheet based on layout
    if sprite_layout == "horizontal":
        # Horizontal sprite sheet for all 6 views
        sheet_width = len(processed_views) * frame_size + (len(processed_views) - 1) * spacing
        sheet_height = frame_size
        sprite_sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
        
        x_offset = 0
        for i, (view, mask) in enumerate(processed_views):
            sprite_sheet.paste(view, (x_offset, 0), mask)
            x_offset += frame_size + spacing
    
    elif sprite_layout == "vertical":
        # Vertical sprite sheet for all 6 views
        sheet_width = frame_size
        sheet_height = len(processed_views) * frame_size + (len(processed_views) - 1) * spacing
        sprite_sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
        
        y_offset = 0
        for i, (view, mask) in enumerate(processed_views):
            sprite_sheet.paste(view, (0, y_offset), mask)
            y_offset += frame_size + spacing
    
    elif sprite_layout == "grid":
        # Grid layout - automatically determine grid size based on number of views
        if len(processed_views) <= 4:
            cols = 2
        else:
            cols = 3  # Use 3 columns for 6 views
        
        rows = math.ceil(len(processed_views) / cols)
        sheet_width = cols * frame_size + (cols - 1) * spacing
        sheet_height = rows * frame_size + (rows - 1) * spacing
        sprite_sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
        
        for i, (view, mask) in enumerate(processed_views):
            row = i // cols
            col = i % cols
            x = col * (frame_size + spacing)
            y = row * (frame_size + spacing)
            sprite_sheet.paste(view, (x, y), mask)
    
    # In the create_sprite_sheet function, update the 2x3_grid section:
    # In the create_game_metadata function, update the 2x3_grid section:
    elif sprite_layout == "2x3_grid":
        # Define grid positions for 2x3 layout based on your arrangement
        grid_positions = [
            (0, 0),  # front
            (1, 0),  # right
            (2, 0),  # back
            (0, 1),  # left
            (1, 1),  # top
            (2, 1)   # bottom
        ]
        
        for i, view_name in enumerate(available_views):
            if i < len(grid_positions):
                col, row = grid_positions[i]
                frame_metadata = {
                    "name": view_name,
                    "index": i,
                    "x": col * (frame_size + spacing),
                    "y": row * (frame_size + spacing),
                    "width": frame_size,
                    "height": frame_size,
                    "pivot": {"x": 0.5, "y": 0.5},
                    "description": views_dict[view_name]["description"]
                }
                metadata["frames"].append(frame_metadata)
    
    return sprite_sheet


def create_game_metadata(sprite_sheet, views_dict, sprite_layout, frame_size, spacing):
    """
    Create metadata for game engine integration
    
    Returns:
        Dictionary with sprite sheet metadata
    """
    metadata = {
        "version": "1.0",
        "generator": "Zero123++ Sprite Sheet Generator",
        "sprite_sheet": {
            "width": sprite_sheet.width,
            "height": sprite_sheet.height,
            "frame_width": frame_size,
            "frame_height": frame_size,
            "frame_count": len(views_dict),  # Now shows actual number of selected views
            "layout": sprite_layout,
            "spacing": spacing,
            "background_removed": True,
            "transparent": True
        },
        "frames": [],
        "game_engine_ready": True,
        "supported_engines": ["Unity", "Unreal Engine", "Godot", "Custom"]
    }
    
    # Define order for all possible views
    all_view_order = ["front", "right", "back", "left", "top", "bottom"]
    available_views = [v for v in all_view_order if v in views_dict]
    
    # Add metadata for each frame based on layout
    if sprite_layout == "horizontal":
        for i, view_name in enumerate(available_views):
            frame_metadata = {
                "name": view_name,
                "index": i,
                "x": i * (frame_size + spacing),
                "y": 0,
                "width": frame_size,
                "height": frame_size,
                "pivot": {"x": 0.5, "y": 0.5},
                "description": views_dict[view_name]["description"]
            }
            metadata["frames"].append(frame_metadata)
    
    elif sprite_layout == "vertical":
        for i, view_name in enumerate(available_views):
            frame_metadata = {
                "name": view_name,
                "index": i,
                "x": 0,
                "y": i * (frame_size + spacing),
                "width": frame_size,
                "height": frame_size,
                "pivot": {"x": 0.5, "y": 0.5},
                "description": views_dict[view_name]["description"]
            }
            metadata["frames"].append(frame_metadata)
    
    elif sprite_layout == "grid":
        # Determine grid dimensions
        if len(available_views) <= 4:
            cols = 2
        else:
            cols = 3
        
        for i, view_name in enumerate(available_views):
            row = i // cols
            col = i % cols
            frame_metadata = {
                "name": view_name,
                "index": i,
                "x": col * (frame_size + spacing),
                "y": row * (frame_size + spacing),
                "width": frame_size,
                "height": frame_size,
                "pivot": {"x": 0.5, "y": 0.5},
                "description": views_dict[view_name]["description"]
            }
            metadata["frames"].append(frame_metadata)
    
    elif sprite_layout == "2x3_grid":
        # Define grid positions for 2x3 layout
        grid_positions = [
            (0, 0),  # front
            (1, 0),  # top
            (2, 0),  # right
            (0, 1),  # left
            (1, 1),  # back
            (2, 1)   # bottom
        ]
        
        for i, view_name in enumerate(available_views):
            if i < len(grid_positions):
                col, row = grid_positions[i]
                frame_metadata = {
                    "name": view_name,
                    "index": i,
                    "x": col * (frame_size + spacing),
                    "y": row * (frame_size + spacing),
                    "width": frame_size,
                    "height": frame_size,
                    "pivot": {"x": 0.5, "y": 0.5},
                    "description": views_dict[view_name]["description"]
                }
                metadata["frames"].append(frame_metadata)
    
    return metadata


def download_sprite_sheet_with_metadata(sprite_sheet, metadata, layout_name):
    """Create download package with sprite sheet and metadata"""
    
    # Create a zip file in memory
    from zipfile import ZipFile
    zip_buffer = BytesIO()
    
    with ZipFile(zip_buffer, 'w') as zip_file:
        # Add sprite sheet
        sprite_buffer = BytesIO()
        sprite_sheet.save(sprite_buffer, format="PNG")
        zip_file.writestr(f"sprite_sheet_{layout_name}.png", sprite_buffer.getvalue())
        
        # Add metadata as JSON
        metadata_json = json.dumps(metadata, indent=2)
        zip_file.writestr(f"metadata_{layout_name}.json", metadata_json)
        
        # Add README file
        readme_content = f"""
# Sprite Sheet Package

This package contains automatically generated sprite sheets and metadata for game engine integration.

## Files:
- sprite_sheet_{layout_name}.png: The sprite sheet image with transparent background
- metadata_{layout_name}.json: Frame metadata for game engine integration

## Game Engine Integration:

### Unity:
1. Import the sprite sheet as a Sprite (2D and UI)
2. Set 'Sprite Mode' to Multiple
3. Use the Sprite Editor and slice using the frame dimensions from metadata
4. Use the frame names for animation states

### For 6-view character sprites:
- Frames are ordered: front, right, back, left, top, bottom
- Perfect for 360-degree character rotation
- Use with animation controllers for smooth rotation

### Unreal Engine:
1. Import as Texture2D
2. Create flipbook animation using frame coordinates from metadata
3. Use frame indices for animation blueprint

### Godot:
1. Import as SpriteFrames resource
2. Create animation using frame coordinates
3. Use frame names for animation states

### Metadata Structure:
- Each frame has x, y coordinates and dimensions
- Pivot point is centered for each frame

Generated by Zero123++ Sprite Sheet Generator
"""
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer


def download_single_image(img, view_name, remove_bg=False):
    """Create download button for a single image"""
    try:
        if remove_bg:
            img = remove_background_single(img)
        else:
            # Ensure proper format for download
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        
        buf = BytesIO()
        img.save(buf, format="PNG")
        
        clean_name = view_name.lower().replace(' ', '_')
        st.download_button(
            label=f"Download {view_name}",
            data=buf.getvalue(),
            file_name=f"orthographic_{clean_name}.png",
            mime="image/png",
            key=f"dl_{clean_name}"
        )
    except Exception as e:
        st.error(f"Error preparing download for {view_name}: {e}")


def calculate_ssim_for_sprite_sheet(sprite_sheet, frame_size, spacing, layout):
    """
    Calculate SSIM between consecutive frames in a sprite sheet to evaluate consistency
    """
    try:
        # Convert sprite sheet to numpy array
        sprite_array = numpy.array(sprite_sheet)
        
        # Extract individual frames based on layout
        frames = []
        
        if layout == "horizontal":
            # Extract frames horizontally
            for i in range(sprite_sheet.width // (frame_size + spacing) + 1):
                x = i * (frame_size + spacing)
                if x + frame_size <= sprite_sheet.width:
                    frame = sprite_array[0:frame_size, x:x+frame_size]
                    frames.append(frame)
        
        elif layout == "vertical":
            # Extract frames vertically
            for i in range(sprite_sheet.height // (frame_size + spacing) + 1):
                y = i * (frame_size + spacing)
                if y + frame_size <= sprite_sheet.height:
                    frame = sprite_array[y:y+frame_size, 0:frame_size]
                    frames.append(frame)
        
        elif layout in ["grid", "2x3_grid"]:
            # Extract frames from grid
            if layout == "2x3_grid":
                cols = 3
            else:
                # Determine columns based on sprite sheet width
                cols = sprite_sheet.width // (frame_size + spacing)
                if cols == 0:
                    cols = 1
            
            rows = sprite_sheet.height // (frame_size + spacing)
            
            for row in range(rows):
                for col in range(cols):
                    x = col * (frame_size + spacing)
                    y = row * (frame_size + spacing)
                    if x + frame_size <= sprite_sheet.width and y + frame_size <= sprite_sheet.height:
                        frame = sprite_array[y:y+frame_size, x:x+frame_size]
                        frames.append(frame)
        
        if len(frames) < 2:
            return {"error": "Not enough frames for SSIM comparison"}
        
        # Calculate SSIM between consecutive frames
        ssim_scores = []
        comparisons = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale for SSIM
            frame1_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGBA2GRAY)
            frame2_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_RGBA2GRAY)
            
            # Ensure same dimensions
            min_height = min(frame1_gray.shape[0], frame2_gray.shape[0])
            min_width = min(frame1_gray.shape[1], frame2_gray.shape[1])
            
            frame1_gray = frame1_gray[:min_height, :min_width]
            frame2_gray = frame2_gray[:min_height, :min_width]
            
            # Calculate SSIM
            score, _ = ssim(frame1_gray, frame2_gray, full=True)
            ssim_scores.append(score)
            
            comparisons.append({
                "frames": (i, i+1),
                "ssim_score": round(score, 4),
                "interpretation": interpret_ssim_score(score)
            })
        
        # Calculate overall statistics
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
        min_ssim = np.min(ssim_scores) if ssim_scores else 0
        max_ssim = np.max(ssim_scores) if ssim_scores else 0
        
        return {
            "frame_count": len(frames),
            "comparisons": comparisons,
            "overall_analysis": {
                "average_ssim": round(avg_ssim, 4),
                "min_ssim": round(min_ssim, 4),
                "max_ssim": round(max_ssim, 4),
                "consistency_rating": get_consistency_rating(avg_ssim),
                "quality_assessment": assess_sprite_quality(avg_ssim, min_ssim)
            }
        }
        
    except Exception as e:
        return {"error": f"SSIM calculation failed: {str(e)}"}


def interpret_ssim_score(score):
    """Interpret what the SSIM score means"""
    if score >= 0.9:
        return "Excellent consistency - nearly identical frames"
    elif score >= 0.8:
        return "Good consistency - very similar frames"
    elif score >= 0.7:
        return "Moderate consistency - noticeable but acceptable differences"
    elif score >= 0.6:
        return "Fair consistency - significant differences"
    else:
        return "Poor consistency - frames are very different"


def get_consistency_rating(avg_ssim):
    """Get a simple rating for sprite sheet consistency"""
    if avg_ssim >= 0.9:
        return "Excellent"
    elif avg_ssim >= 0.8:
        return "Good"
    elif avg_ssim >= 0.7:
        return "Moderate"
    elif avg_ssim >= 0.6:
        return "Fair"
    else:
        return "Poor"


def assess_sprite_quality(avg_ssim, min_ssim):
    """Provide overall quality assessment"""
    if avg_ssim >= 0.8 and min_ssim >= 0.7:
        return "High quality - suitable for professional use"
    elif avg_ssim >= 0.7 and min_ssim >= 0.6:
        return "Good quality - suitable for most game applications"
    elif avg_ssim >= 0.6:
        return "Acceptable quality - may need manual adjustments"
    else:
        return "Low quality - consider regenerating or manual editing"


def create_quality_report(ssim_results, metadata):
    """Create a comprehensive quality report"""
    if "error" in ssim_results:
        return {"error": ssim_results["error"]}
    
    report = {
        "quality_report": {
            "generated_at": str(np.datetime64('now')),
            "sprite_sheet_info": {
                "dimensions": f"{metadata['sprite_sheet']['width']}x{metadata['sprite_sheet']['height']}",
                "frame_size": metadata['sprite_sheet']['frame_width'],
                "frame_count": metadata['sprite_sheet']['frame_count'],
                "layout": metadata['sprite_sheet']['layout']
            },
            "ssim_analysis": ssim_results["overall_analysis"],
            "frame_comparisons": ssim_results["comparisons"],
            "recommendations": generate_recommendations(ssim_results)
        }
    }
    
    return report


def generate_recommendations(ssim_results):
    """Generate recommendations based on SSIM analysis"""
    if "error" in ssim_results:
        return ["Unable to generate recommendations due to analysis error"]
    
    overall = ssim_results["overall_analysis"]
    recommendations = []
    
    if overall["average_ssim"] < 0.7:
        recommendations.append("Consider regenerating with different parameters for better consistency")
        recommendations.append("Try increasing inference steps for more stable results")
        recommendations.append("Check input image quality and background removal")
    
    if overall["min_ssim"] < 0.6:
        recommendations.append("Some frames have significant differences - may affect animation smoothness")
        recommendations.append("Consider manual adjustment of inconsistent frames")
    
    if overall["average_ssim"] >= 0.8:
        recommendations.append("Excellent consistency - ready for game engine integration")
        recommendations.append("No additional processing needed")
    
    # Add general recommendations
    recommendations.extend([
        "For character sprites, aim for SSIM > 0.7 for smooth animation",
        "Higher CFG scale values may improve consistency but reduce variety",
        "Ensure consistent lighting in input image for best results"
    ])
    
    return recommendations


@st.cache_data
def check_dependencies():
    reqs = []
    try:
        import diffusers
    except ImportError:
        import traceback
        traceback.print_exc()
        print("Error: `diffusers` not found.", file=sys.stderr)
        reqs.append("diffusers==0.20.2")
    else:
        if not diffusers.__version__.startswith("0.20"):
            print(
                f"Warning: You are using an unsupported version of diffusers ({diffusers.__version__}), which may lead to performance issues.",
                file=sys.stderr
            )
            print("Recommended version is `diffusers==0.20.2`.", file=sys.stderr)
    try:
        import transformers
    except ImportError:
        import traceback
        traceback.print_exc()
        print("Error: `transformers` not found.", file=sys.stderr)
        reqs.append("transformers==4.29.2")
    if torch.__version__ < '2.0':
        try:
            import xformers
        except ImportError:
            print("Warning: You are using PyTorch 1.x without a working `xformers` installation.", file=sys.stderr)
            print("You may see a significant memory overhead when running the model.", file=sys.stderr)
    if len(reqs):
        print(f"Info: Fix all dependency errors with `pip install {' '.join(reqs)}`.")


@st.cache_resource
def load_zero123plus_pipeline():
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    
    # Use the local model directory
    local_model_path = "./zero123plus-local"
    custom_pipeline_path = "./zero123plus-local/zero123plus_pipeline/pipeline.py"
    
    if not os.path.exists(local_model_path):
        st.error(f"Local model directory not found: {local_model_path}")
        st.info("Please make sure the zero123plus-local folder exists in the current directory.")
        return None
    
    if not os.path.exists(custom_pipeline_path):
        st.error(f"Custom pipeline not found: {custom_pipeline_path}")
        st.info("Please make sure the pipeline.py file exists in the zero123plus-local/zero123plus_pipeline directory.")
        return None
    
    try:
        # Load from local directory with custom pipeline
        st.info("Loading Zero123++ with custom pipeline...")
        pipeline = DiffusionPipeline.from_pretrained(
            local_model_path,
            custom_pipeline=custom_pipeline_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True  # Force using local files only
        )
        
        st.success("Successfully loaded Zero123++ model with custom pipeline!")
        
    except Exception as e:
        st.error(f"Failed to load with custom pipeline: {e}")
        st.info("Trying to load without custom pipeline...")
        
        try:
            # Fallback: Load without custom pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True
            )
            st.success("Successfully loaded Zero123++ model (without custom pipeline)!")
        except Exception as e2:
            st.error(f"Failed to load model: {e2}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    # Configure scheduler
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    
    # Move to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Move components to device
    pipeline.unet = pipeline.unet.to(device)
    pipeline.vae = pipeline.vae.to(device)
    
    # Handle vision encoder (might be called image_encoder or vision_encoder)
    if hasattr(pipeline, 'vision_encoder'):
        pipeline.vision_encoder = pipeline.vision_encoder.to(device)
    elif hasattr(pipeline, 'image_encoder'):
        pipeline.image_encoder = pipeline.image_encoder.to(device)
    
    # Handle text encoder
    if hasattr(pipeline, 'text_encoder'):
        pipeline.text_encoder = pipeline.text_encoder.to(device)
    
    # Set device attribute for generator
    pipeline._device = device
    
    sys.main_lock = threading.Lock()
    return pipeline


check_dependencies()

# Load pipeline
pipeline = load_zero123plus_pipeline()

if pipeline is None:
    st.error("Failed to load pipeline. Please check the model files.")
    st.stop()

SAMAPI.get_instance()
torch.set_grad_enabled(False)

st.title("🎯 Zero123++ - Advanced Sprite Sheet Generator")
st.success("Using locally downloaded Zero123++ model with custom pipeline!")
st.info("🔄 Generates orthographic views and automatically creates game-ready sprite sheets with metadata!")

# Sprite Sheet Configuration
st.sidebar.header("🎮 Sprite Sheet Settings")

# Sprite layout options
sprite_layout = st.sidebar.selectbox(
    "Sprite Sheet Layout:",
    ["horizontal", "vertical", "grid", "2x3_grid"],
    index=0,
    help="""Choose how to arrange frames in the sprite sheet:
    - horizontal: All frames in one row
    - vertical: All frames in one column  
    - grid: Automatic grid layout
    - 2x3_grid: Perfect for 6 views (2 rows, 3 columns)"""
)

frame_size = st.sidebar.slider(
    "Frame Size:",
    min_value=64,
    max_value=512,
    value=256,
    step=64,
    help="Size of each frame in the sprite sheet"
)

spacing = st.sidebar.slider(
    "Frame Spacing:",
    min_value=0,
    max_value=50,
    value=10,
    help="Spacing between frames in the sprite sheet"
)

# View selection for sprite sheets
st.sidebar.subheader("🔄 View Selection")
available_views = ["front", "right", "back", "left", "top", "bottom"]
selected_views = st.sidebar.multiselect(
    "Select Views for Sprite Sheet:",
    options=available_views,
    default=available_views,  # Default to all 6 views
    help="Choose which views to include in sprite sheets. Select all 6 for complete character sprite sheet."
)

# Layout options for individual view display
st.sidebar.subheader("🎨 Display Options")
columns_layout = st.sidebar.selectbox(
    "Layout Columns:",
    [2, 3, 4, 6],
    index=1,
    help="Number of columns for displaying individual views"
)

# Main app interface
prog = st.progress(0.0, "Idle")
pic = st.file_uploader("Upload an Image", key='imageinput', type=['png', 'jpg', 'webp'])

left, right = st.columns(2)
with left:
    rem_input_bg = st.checkbox("Remove Input Background")
with right:
    rem_output_bg = st.checkbox("Remove Output Background", value=True)

num_inference_steps = st.slider("Number of Inference Steps", 15, 100, 75)
cfg_scale = st.slider("Classifier Free Guidance Scale", 1.0, 10.0, 4.0)
seed = st.text_input("Seed", "42")

submit = st.button("Generate Sprite Sheets")

results_container = st.container()

with results_container:
    if pic is not None and submit:
        prog.progress(0.03, "Waiting in Queue...")
        with sys.main_lock:
            seed = int(seed)
            img = Image.open(pic)
            
            if max(img.size) > 1280:
                w, h = img.size
                w = round(1280 / max(img.size) * w)
                h = round(1280 / max(img.size) * h)
                img = img.resize((w, h))
            
            left, right = st.columns(2)
            with left:
                st.image(img)
                st.caption("Input Image")
            
            prog.progress(0.1, "Preparing Inputs")
            if rem_input_bg:
                with right:
                    img = segment_img(img)
                    st.image(img)
                    st.caption("Input (Background Removed)")
            
            img = expand2square(img, (127, 127, 127, 0))
            
            st.info("🚀 Generating 6 Orthographic Views...")
            st.info(f"📊 Device: {pipeline._device} | Steps: {num_inference_steps} | Guidance: {cfg_scale}")
            
            # Create a more detailed progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Custom callback to track progress
            def progress_callback(step, timestep, latents):
                progress = 0.1 + 0.8 * (step / num_inference_steps)
                progress_placeholder.progress(progress)
                status_placeholder.text(f"🔄 Diffusion step {step+1}/{num_inference_steps}")
            
            # Update status before starting
            progress_placeholder.progress(0.1)
            status_placeholder.text("🔄 Starting diffusion process...")
            
            # Disable progress bar to avoid conflicts
            if hasattr(pipeline, 'set_progress_bar_config'):
                pipeline.set_progress_bar_config(disable=True)
            
            try:
                # Call the pipeline with progress callback
                result = pipeline(
                    img,  # PIL Image
                    num_inference_steps=num_inference_steps,
                    guidance_scale=cfg_scale,
                    generator=torch.Generator(device=pipeline._device).manual_seed(seed),
                    callback=progress_callback
                )
                
                # Clear the progress placeholders
                progress_placeholder.empty()
                status_placeholder.empty()
                
                # Handle different return types
                if hasattr(result, 'images'):
                    result_image = result.images[0]
                elif isinstance(result, tuple) and len(result) > 0:
                    result_image = result[0]  # Assume first element is the image
                elif isinstance(result, list) and len(result) > 0:
                    result_image = result[0]
                else:
                    st.error(f"Unexpected result type: {type(result)}")
                    result_image = None
                
                if result_image is not None:
                    prog.progress(0.9, "Creating Sprite Sheets and Metadata...")
                    
                    # Separate all 6 views
                    all_views = separate_orthographic_views(result_image)
                    
                    # Create custom layout with selected views only
                    custom_views = create_custom_view_layout(all_views, selected_views)
                    
                    if custom_views:
                        # Display individual views
                        st.subheader("🎯 Generated Orthographic Views")
                        
                        view_names = list(custom_views.keys())
                        cols = st.columns(columns_layout)
                        
                        for i, view_name in enumerate(view_names):
                            view_data = custom_views[view_name]
                            col_idx = i % columns_layout
                            
                            with cols[col_idx]:
                                display_img = view_data["image"]
                                if rem_output_bg:
                                    display_img = remove_background_single(display_img)
                                else:
                                    # Ensure proper format for display
                                    if display_img.mode != 'RGB':
                                        display_img = display_img.convert('RGB')
                                
                                st.image(display_img, use_column_width=True)
                                st.caption(f"**{view_data['name']}**")
                                st.write(f"*{view_data['description']}*")
                        
                        # Create and display sprite sheets
                        st.subheader("🔄 Generated Sprite Sheets")
                        
                        # Create sprite sheet
                        sprite_sheet = create_sprite_sheet(
                            custom_views, 
                            sprite_layout, 
                            frame_size, 
                            spacing
                        )
                        
                        if sprite_sheet:
                            # Display sprite sheet
                            st.image(sprite_sheet, use_column_width=True)
                            st.caption(f"Sprite Sheet Layout: {sprite_layout.upper()} | Frame Size: {frame_size}px | Spacing: {spacing}px")
                            
                            # Create metadata
                            metadata = create_game_metadata(
                                sprite_sheet, 
                                custom_views, 
                                sprite_layout, 
                                frame_size, 
                                spacing
                            )
                            
                            # Quality Evaluation Section
                            st.subheader("📊 Sprite Sheet Quality Evaluation")
                            
                            # Calculate SSIM metrics
                            ssim_results = calculate_ssim_for_sprite_sheet(
                                sprite_sheet, frame_size, spacing, sprite_layout  # Added sprite_layout parameter
                            )
                            
                            if "error" not in ssim_results:
                                # Display quality metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    avg_ssim = ssim_results["overall_analysis"]["average_ssim"]
                                    st.metric(
                                        "Average SSIM Score", 
                                        f"{avg_ssim:.3f}",
                                        help="Structural Similarity Index (0-1, higher is better)"
                                    )
                                
                                with col2:
                                    consistency = ssim_results["overall_analysis"]["consistency_rating"]
                                    st.metric(
                                        "Consistency Rating", 
                                        consistency,
                                        help="How consistent the frames are with each other"
                                    )
                                
                                with col3:
                                    quality = ssim_results["overall_analysis"]["quality_assessment"]
                                    st.metric(
                                        "Quality Assessment", 
                                        quality.split(" - ")[0],
                                        help="Overall quality assessment for game use"
                                    )
                                
                                # Detailed analysis
                                with st.expander("📈 Detailed SSIM Analysis"):
                                    st.write("**Frame-to-Frame Comparisons:**")
                                    for comp in ssim_results["comparisons"]:
                                        st.write(f"Frames {comp['frames'][0]}→{comp['frames'][1]}: SSIM = {comp['ssim_score']} - {comp['interpretation']}")
                                    
                                    st.write("**Overall Statistics:**")
                                    st.json(ssim_results["overall_analysis"])
                                
                                # Generate and display quality report
                                quality_report = create_quality_report(ssim_results, metadata)
                                
                                with st.expander("📋 Quality Report & Recommendations"):
                                    st.write("**Quality Report:**")
                                    st.json(quality_report["quality_report"])
                                    
                                    st.write("**Recommendations:**")
                                    for i, recommendation in enumerate(quality_report["quality_report"]["recommendations"]):
                                        st.write(f"{i+1}. {recommendation}")
                                
                            else:
                                st.warning(f"Could not calculate quality metrics: {ssim_results['error']}")
                                quality_report = {"error": ssim_results["error"]}
                            
                            # Display metadata
                            with st.expander("📋 Sprite Sheet Metadata"):
                                st.json(metadata)
                            
                            # Download section
                            st.subheader("📥 Download Assets")
                            
                            # Create download package
                            zip_buffer = download_sprite_sheet_with_metadata(
                                sprite_sheet, metadata, sprite_layout
                            )
                            
                            st.download_button(
                                label="📦 Download Complete Sprite Sheet Package (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"sprite_sheet_package_{sprite_layout}.zip",
                                mime="application/zip",
                                key="dl_sprite_package"
                            )
                        
                        # Show all generated views in expander
                        with st.expander("🔍 Show All 6 Generated Views"):
                            st.write("All views generated by Zero123++:")
                            all_cols = st.columns(3)
                            all_view_names = ["Front", "Right", "Back", "Left", "Top", "Bottom"]
                            for i, view_img in enumerate(all_views):
                                col_idx = i % 3
                                with all_cols[col_idx]:
                                    display_view = view_img
                                    if display_view.mode != 'RGB':
                                        display_view = display_view.convert('RGB')
                                    st.image(display_view, use_column_width=True)
                                    st.caption(f"**{all_view_names[i]} View**")
                    
                    prog.progress(1.0, "Complete!")
                else:
                    st.error("No image was generated.")
                
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
                import traceback
                st.code(traceback.format_exc())