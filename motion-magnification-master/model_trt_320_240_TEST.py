import torch
import torch_tensorrt
import cv2
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from deepmag import dataset, viz, model, train
from jetson_utils import videoSource, videoOutput, cudaImage
import time

camera = videoSource("/dev/video0", ["-input-width=320", "-input-height=240"])      # '/dev/video0' for V4L2
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

m = torch.jit.load('model_trt_320_240.ts').to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

amp_f = 5
amp_f_tensor = torch.tensor([[float(amp_f)]], dtype=torch.float, device=device)

#frame_a = cv2.imread('xa.jpg')[...,::-1]
#frame_a = torch.unsqueeze(transform(to_pil_image(frame_a)).to(device), 0)
#frame_b = cv2.imread('xb.jpg')[...,::-1]
#frame_b = torch.unsqueeze(transform(to_pil_image(frame_b)).to(device), 0)

print("capturing frame a")
frame_a = camera.Capture()
print("waiting 0.1s...")
time.sleep(0.1)
print("capturing frame b")
frame_b = camera.Capture()
example_input_0 = torch.unsqueeze(torch.as_tensor(frame_a, device=device).permute(2, 0, 1), 0).contiguous()/255
example_input_1 = torch.unsqueeze(torch.as_tensor(frame_b, device=device).permute(2, 0, 1), 0).contiguous()/255

print(frame_a.shape)
print(type(frame_a))

pred, _, _ = m.forward(
    example_input_0,
    example_input_1,
    amp_f_tensor.reshape(1, 1))
    
torchvision.utils.save_image(example_input_0, "example_input_0.jpg")
torchvision.utils.save_image(example_input_1, "example_input_1.jpg")
    
cv2.imwrite("pred_live.jpg", pred[0].permute(1,2,0).cpu().detach().numpy()[...,::-1]*255)
