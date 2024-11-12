import os
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from bin.imageScales import imageScale, imageGridCreator
from model import model_
from bin.dataset import loadTestds, loadDs, mean, std, padsize
from datetime import datetime

def img_to_np(input, H, W):
    image = input[:, :, padsize:H - padsize, padsize:W - padsize].clone()
    input_numpy = image[:, :, :H, :W].clone().cpu().numpy().reshape(3, H - padsize * 2, W - padsize * 2).transpose(1, 2, 0)
    for i in range(3):
        input_numpy[:, :, i] = input_numpy[:, :, i] * std[i] + mean[i]
    return input_numpy

# Argument parser for folder input
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="defaultDataset", help="Name of the folder containing images")
opt = parser.parse_args()

# Create output directory
output_path = f"images/{opt.dir}/output"
dataset_path = f"images/{opt.dir}/"
os.makedirs(output_path, exist_ok=True)

# Set device
device = 'cpu'

# Initialize generator model
Generator = model_().to(device)
Generator.eval()
Generator.load_state_dict(torch.load("trained_model_v4.pth", map_location=device))

# Check if ground truth is available
ground_truth_available = False
if os.path.exists(f"images/{opt.dir}/gt") and \
   len(os.listdir(f"images/{opt.dir}/input")) == len(os.listdir(f"images/{opt.dir}/gt")):
    ground_truth_available = True

# Load dataset
test_dataset = loadTestds(f"images/{opt.dir}") if ground_truth_available else loadDs(f"images/{opt.dir}")

# Initialize variables for PSNR/SSIM calculations
all_psnr = 0.0
all_ssim = 0.0
html_report = "<html><head><style>body{font-family:Arial,sans-serif;background-color:#f4f4f9;color:#333;margin:0;padding:0;}h1{text-align:center;color:#4CAF50;margin-top:20px;}table{width:80%;margin:20px auto;border-collapse:collapse;background-color:#fff;box-shadow:0 2px 10px rgba(0,0,0,0.1);}th,td{padding:15px;text-align:center;border:1px solid #ddd;}th{background-color:#4CAF50;color:white;font-size:18px;}tr:nth-child(even){background-color:#f9f9f9;}tr:hover{background-color:#f1f1f1;}.image-cell img{max-width:100px;height:auto;border-radius:5px;}</style></head><body><h1>Image Processing Results</h1><table><tr><th>Image Name</th><th>Input</th><th>Output</th></tr>"

# Process images
for image_num in tqdm(range(len(test_dataset))):
    data = test_dataset[image_num]
    R = data["R"].to(device)

    # Pad and reshape image
    _, first_h, first_w = R.size()
    R = torch.nn.functional.pad(R, (0, (R.size(2) // 16) * 16 + 16 - R.size(2), 0, (R.size(1) // 16) * 16 + 16 - R.size(1)), "constant")
    R = R.view(1, 3, R.size(1), R.size(2))

    # Generate output using the model
    with torch.no_grad():
        output = Generator(R)

    output_np = np.clip(img_to_np(output, first_h, first_w) + 0.015, 0, 1)
    R_np = img_to_np(R, first_h, first_w)
    final_output = np.fmin(output_np, R_np)

    # Save output image
    output_image_filename = f"{data['Name']}_output.png"
    output_image_path = os.path.join(output_path, output_image_filename)
    Image.fromarray(np.uint8(final_output * 255)).save(output_image_path)

    # Save input image for HTML display
    input_image_filename = f"{data['Name']}_input.png"
    input_image_path = os.path.join(output_path, input_image_filename)
    Image.fromarray(np.uint8(R_np * 255)).save(input_image_path)

    # Calculate PSNR/SSIM if ground truth is available
    psnr_value, ssim_value = "N/A", "N/A"
    if ground_truth_available:
        B = data["B"].astype(np.float32)
        psnr_value = psnr(B, final_output.astype(np.float32))
        ssim_value = ssim(B, final_output.astype(np.float32), multichannel=True)
        all_psnr += psnr_value
        all_ssim += ssim_value

    # Add entry to HTML content
    html_report += f"<tr><td>{data['Name']}</td>"
    html_report += f"<td><img src='output/{input_image_filename}' width='256'></td>"
    html_report += f"<td><img src='output/{output_image_filename}' width='256'></td>"

# Calculate average PSNR and SSIM
if ground_truth_available:
    avg_psnr = all_psnr / len(test_dataset)
    avg_ssim = all_ssim / len(test_dataset)
    html_report += f"<tr><td colspan='3'>Average</td><td>{avg_psnr:.2f}</td><td>{avg_ssim:.3f}</td></tr>"

html_report += "</table></body></html>"

# Save HTML file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
html_file_path = os.path.join(dataset_path, f"results_{timestamp}.html")
with open(html_file_path, "w") as f:
    f.write(html_report)

print(f"HTML results saved to {html_file_path}")

import webbrowser
import os

# Check if the file exists before opening
if os.path.exists(html_file_path):
    # Open the HTML file in the default web browser
    webbrowser.open(f'file://{os.path.abspath(html_file_path)}')
else:
    print(f"Error: {html_file_path} does not exist.")
