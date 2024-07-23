#!/usr/bin/env python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

from jetson_utils import videoSource, videoOutput, cudaImage
import torch

camera = videoSource("/dev/video0", ["-input-width=320", "-input-height=240"])      # '/dev/video0' for V4L2
display = videoOutput("display://0") # 'my_video.mp4' for file
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

m = torch.load('data/models/20191202-b10-r0.1-lr0.0001-00.pt').to(device).eval()
amp_f = 5
amp_f_tensor = torch.tensor([[float(amp_f)]], dtype=torch.float, device=device)

#prev_tensor = torch.unsqueeze(torch.as_tensor(camera.Capture(), device=device).permute(2, 0, 1), 0).float()

while display.IsStreaming():
    frame = camera.Capture()
    tensor = torch.unsqueeze(torch.as_tensor(frame, device=device).permute(2, 0, 1), 0).float()

    if frame is None: # capture timeout
        continue
    
    torch.cuda.empty_cache()
    
    #pred, _, _ = m.forward(tensor, prev_tensor, amp_f_tensor)
    #prev_tensor = pred
    #pred = pred[0].permute(2, 1, 0)
    tensor = tensor.permute(0, 2, 3, 1)
    cuda_pred = cudaImage(ptr=tensor.data_ptr(), width=tensor.shape[2], height=tensor.shape[1], format="rgb32f")
    
    display.Render(cuda_pred)
    display.SetStatus("VMM")
