from Data_generation import generate_examples
from CNN import StarDetector
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from Data_generation import iou
from tqdm import tqdm
import os

os.chdir("Circle_detection")

def find_circle(img):
    model = StarDetector()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load('circle_detection.pth',map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        image = np.expand_dims(np.asarray(img), axis=0)
        image = torch.from_numpy(np.array(image, dtype=np.float32))
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize (image)
        image = image.unsqueeze(0)
        #image = img.float()
        output = model(image)


    return [round(i) for i in (200*output).tolist()[0]]

def main():
    results = []
    example_generator = generate_examples(img_size=224)
    for _ in tqdm(range(500)):
        img, param = next(example_generator)
        # img=img.to(device)
        # param=param.to(device)
        detected = find_circle(img)
        results.append(iou(param, detected))
    results = np.array(results)
    print("The IOU Score is ",(results > 0.7).mean(),end="")

main()
