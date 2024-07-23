from jetson_utils import videoSource, videoOutput, cudaImage, cudaDeviceSynchronize
import torch
import torch_tensorrt
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from deepmag import dataset, viz, model, train

# generate trt model for resolution 320x240

camera = videoSource("/dev/video0", ["-input-width=320", "-input-height=240"])      # '/dev/video0' for V4L2
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

m = torch.load('data/models/20191202-b10-r0.1-lr0.0001-00.pt').to(device).eval()

frame = camera.Capture()
example_input_0 = torch.unsqueeze(torch.as_tensor(frame, device=device).permute(2, 0, 1), 0)/255
example_input_1 = torch.unsqueeze(torch.as_tensor(frame, device=device).permute(2, 0, 1), 0)/255
example_input_2 = torch.tensor([[float(5)]], dtype=torch.float, device=device)

print("tracing model")
model_traced = torch.jit.trace(m, (example_input_0, example_input_1, example_input_2))
print("done tracing model")

torch.cuda.empty_cache()

print("generating trt model")
trt_model_fp32 = torch_tensorrt.compile(model_traced, inputs = [example_input_0, example_input_1, example_input_2],
    enabled_precisions = torch.float32, # Run with FP32
    workspace_size = 20 << 22
)
print("done generating trt model")

torch.jit.save(trt_model_fp32, "PYTHON3_model_trt_320_240.ts")
