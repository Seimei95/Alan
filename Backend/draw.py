from PIL import Image, ImageDraw
import numpy as np

def draw_forgery_boxes(mask, image_np, save_path):
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)

    visited = set()
    h, w = mask.shape

    for y in range(0, h, 32):
        for x in range(0, w, 32):
            label = mask[y, x]
            if label > 0 and label not in visited:
                visited.add(label)
                coords = np.argwhere(mask == label)
                ys, xs = coords[:, 0], coords[:, 1]
                box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                draw.rectangle(box, outline="red", width=3)

    img.save(save_path)
