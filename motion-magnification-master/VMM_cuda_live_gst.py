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

import jetson.utils
import torch

camera = jetson.utils.GstCamera(160, 120, "/dev/video0")      # '/dev/video0' for V4L2
display = jetson.utils.glDisplay() # 'my_video.mp4' for file
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

m = torch.load('data/models/20191202-b10-r0.1-lr0.0001-00.pt').to(device).eval()
amp_f = 100
amp_f_tensor = torch.tensor([[float(amp_f)]], dtype=torch.float, device=device)
prev_frame = torch.zeros((1, 3, 160, 120)).to(device)

while display.IsOpen():
    img, width, height = camera.CaptureRGBA()
    print(img)

    if frame is None: # capture timeout
        continue
    
    torch.cuda.empty_cache()
    
    pred, _, _ = m.forward(frame, prev_frame, amp_f_tensor)
    
    display.Render(cudaFromNumpy(pred.detach().cpu().numpy()))
    prev_frame = frame
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
