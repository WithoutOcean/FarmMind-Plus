import os
from PIL import Image,ImageDraw
import cv2
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.transform import Affine
from rasterio.transform import xy
def temporal_mark(image_path,temporal_path,bbox,save_dir):
    
    with rasterio.open(image_path) as small_src, rasterio.open(temporal_path) as large_src:
   
        small_transform = small_src.transform
        small_crs = small_src.crs
        small_width, small_height = small_src.width, small_src.height
        
        large_transform = large_src.transform
        large_crs = large_src.crs
        large_width, large_height = large_src.width, large_src.height

        small_top_left = small_transform * (0, 0) 

        row, col = large_src.index(*small_top_left)  

        crop_x1 = col
        crop_y1 = row
        crop_x2 = col + small_width
        crop_y2 = row + small_height

     
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(large_src.width, crop_x2)
        crop_y2 = min(large_src.height, crop_y2)

       
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        if crop_w <= 0 or crop_h <= 0:
            raise ValueError(
                
                f"x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}, "
                f"w={crop_w}, h={crop_h}"
            )

       
        window = rasterio.windows.Window( crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)
        
       
        cropped_data = large_src.read(window=window)
        cropped_data=cropped_data.transpose(1, 2, 0)
      
        cropped_image = Image.fromarray(cropped_data)  
        cropped_image=cropped_image.resize((512, 512))
        cropped_image_ori = cropped_image.copy()
        draw = ImageDraw.Draw(cropped_image)
       
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        image_name=os.path.basename(temporal_path)
        crop_name=f"{bbox}_{image_name}"
         
        ouput_path=os.path.join(save_dir,crop_name.replace(".tif",".png"))
        output_path_ori=os.path.join(save_dir,image_name.replace(".tif",".png"))
        cropped_image.save(ouput_path, format='PNG')
        cropped_image_ori.save(output_path_ori,format="PNG")
    return cropped_image,cropped_image_ori,ouput_path
def context_mark(image_path,context_path,bbox,save_dir):
    new_bbox=[]
    
    with rasterio.open(image_path) as small_src, rasterio.open(context_path) as large_src:
        small_transform = small_src.transform
        small_tf = small_src.transform
        large_transform = large_src.transform
        LW, LH = large_src.width, large_src.height

       
        x1, y1, x2, y2 = map(float, bbox)
        cx_small = (x1 + x2) / 2.0
        cy_small = (y1 + y2) / 2.0

        
        gx, gy = xy(small_transform, cy_small, cx_small, offset='center')

       
        row_c, col_c = large_src.index(gx, gy)  
       
        def fixed_window_around(col_center, row_center, w, h, W, H):
            col0 = int(round(col_center - w // 2))
            row0 = int(round(row_center - h // 2))
           
            col0 = max(0, col0)
            row0 = max(0, row0)
           
            if col0 + w > W:
                col0 = max(0, W - w)
            if row0 + h > H:
                row0 = max(0, H - h)
            return Window(col0, row0, w, h)
        
        
        window_512 = fixed_window_around(col_c, row_c, 512, 512, LW, LH)
        
       
        cropped512_data = large_src.read(window=window_512)
        
        def save_cropped_geotiff(src_dataset, data, window, out_path):
            profile = src_dataset.profile.copy()
            profile.update({
                "driver": "GTiff",
                "width": int(window.width),
                "height": int(window.height),
                "transform": window_transform(window, src_dataset.transform),
                "count": data.shape[0],
                "dtype": data.dtype,
               
                "tiled": False,
                "compress": "LZW"
            })
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)

        base = os.path.splitext(os.path.basename(image_path))[0]
        tif_out_512 = os.path.join(save_dir, f"{base}_crop512.tif")
        
        save_cropped_geotiff(large_src, cropped512_data, window_512, tif_out_512)
    
       
        def small_pt_to_large(px, py):
            gx_, gy_ = xy(small_tf, py, px, offset="center")
            r_big, c_big = large_src.index(gx_, gy_)
            return int(c_big), int(r_big)  # (x, y) = (col, row)
        X1L, Y1L = small_pt_to_large(x1, y1)
        X2L, Y2L = small_pt_to_large(x2, y2)
        xL1, xL2 = sorted((X1L, X2L))
        yL1, yL2 = sorted((Y1L, Y2L))
    
   
        x1c = xL1 - int(window_512.col_off);  y1c = yL1 - int(window_512.row_off)
        x2c = xL2 - int(window_512.col_off);  y2c = yL2 - int(window_512.row_off)
        new_bbox.append([x1c, y1c, x2c, y2c])

        cropped512_data=cropped512_data.transpose(1, 2, 0)
        cropped_image_512 = Image.fromarray(cropped512_data)
        cropped_image_512=cropped_image_512.resize((512, 512))
        cropped_image_512ori = cropped_image_512.copy()
        draw = ImageDraw.Draw(cropped_image_512)
       
        draw.rectangle([new_bbox[-1][0], new_bbox[-1][1], new_bbox[-1][2] ,new_bbox[-1][3] ], outline="red", width=3)
      
     
        image_name=os.path.basename(image_path)
        crop_name512=f"{new_bbox[-1]}_512{image_name}"
        ouput_path512=os.path.join(save_dir,crop_name512.replace(".tif",".png"))
        cropped_image_512.save(ouput_path512, format='PNG')
       

    return cropped_image_512,cropped_image_512ori,ouput_path512,new_bbox
    
def retrieve_temporal(image_path,temporal_dir,crop_temporal_dir,bbox):
    all_temporal_name=os.listdir(temporal_dir)
    crop_temporal_paths=[]
    all_crop_image=[]
    all_crop_ori=[]
    month=[]
    image_name=os.path.basename(image_path).split(".tif")[0] if ".tif" in image_path else os.path.basename(image_path).split(".png")[0]
    name_list=image_name.split("_")
    image_locat=f"{name_list[0]}_{name_list[1]}_{name_list[2]}_{name_list[3]}"

    temporal_locat=[]
    for temporal_img in all_temporal_name:
        tem_name=temporal_img.split(".tif")[0]
        tem_name_list=tem_name.split("_")
        tem_locat=f"{tem_name_list[0]}_{tem_name_list[1]}_{tem_name_list[2]}_{tem_name_list[3]}"
        temporal_locat.append(tem_locat)
    month_ori=name_list[4]
    indices = [idx for idx, element in enumerate(temporal_locat) if image_locat==element]
    for i in indices:
        temporal_name=all_temporal_name[i].split(".tif")[0] if ".tif" in all_temporal_name[i] else all_temporal_name[i].split(".png")[0]
        month_str=temporal_name.split("_")[4]

        if int(month_str)==int(month_ori):
            continue

        one_temporal=os.path.join(temporal_dir,all_temporal_name[i])
        result_data=temporal_mark(image_path,one_temporal,bbox,crop_temporal_dir)
        all_crop_image.append(result_data[0])
        all_crop_ori.append(result_data[1])
        crop_temporal_paths.append(result_data[2])
        month.append(month_str)
    return all_crop_image,all_crop_ori,month

def retrieve_context(image_path,context_dir,crop_context_dir,bbox):
    all_context_name=os.listdir(context_dir)
    
    crop_context_paths=[]
    all_crop_ori=[]
    image_name=os.path.basename(image_path).split("_patch")[0]
    try:
        indices = [idx for idx, element in enumerate(all_context_name) if image_name==element.split(".tif")[0]]
        indices=indices[0]
    except:
        print("not the image")
        print(image_name)
    one_context=os.path.join(context_dir,all_context_name[indices])
    result_data=context_mark(image_path,one_context,bbox,crop_context_dir)
    all_crop_image=[result_data[0]]
    all_crop_ori=[result_data[1]]
    crop_context_paths=[result_data[2]]
    new_bbox=result_data[3][-1]
    
    return all_crop_image,all_crop_ori,new_bbox





