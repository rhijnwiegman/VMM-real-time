import torch
import torch_tensorrt
from deepmag import dataset, viz, model, train

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

m = torch.load('data/models/20191202-b10-r0.1-lr0.0001-00.pt').to(device).eval()

batch_size = (1, 3, 1280, 720)
example_input_0 = torch.randn(batch_size).to(device)
example_input_1 = torch.randn(batch_size).to(device)
example_input_2 = torch.randn((1, 1)).to(device)

model_traced = torch.jit.trace(m, (example_input_0, example_input_1, example_input_2))

torch.cuda.empty_cache()

trt_model_fp32 = torch_tensorrt.compile(model_traced, inputs = [example_input_0, example_input_1, example_input_2], enabled_precisions = torch.float32, workspace_size = 1 << 22)

torch.jit.save(trt_model_fp32, "model_trt_1280_720.ts")
