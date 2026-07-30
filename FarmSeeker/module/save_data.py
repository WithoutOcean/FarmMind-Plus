import numpy as np
import cv2

def save_mask(pred_mask,image_path,save_dir,image_np):
    pred_mask = pred_mask.detach().cpu().numpy()[0]
    pred_mask = pred_mask > 0
    save_path = "{}/{}.png".format(
        save_dir, image_path.split("/")[-1].split(".tif")[0],
    )
    red_layer = np.zeros_like(image_np, dtype=np.float32)
    red_layer[pred_mask] = [0, 255, 0] 
    red_layer = cv2.cvtColor(red_layer, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, red_layer)
    print("{} has been saved.".format(save_path))

    save_path = "{}/{}_img.png".format(
        save_dir, image_path.split("/")[-1].split(".tif")[0]
    )
    save_img = image_np.copy()
    save_img[pred_mask] = (
        image_np * 0.85
        + pred_mask[:, :, None].astype(np.uint8) * np.array([0, 255, 0]) * 0.15
    )[pred_mask]
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_img)
    print("{} has been saved.".format(save_path))
