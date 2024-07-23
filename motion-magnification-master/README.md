## Run (examples)

    # run original model in CUDA mode
    python3 VMM_cuda_live.py

    # run optimized model in TensorRT mode
    python3 VMM_trt_live.py

Note: these models were designed for NVIDIA Jetson Orin Nano Developer Kit hardware using a MICROSOFT LifeCam HD-3000 webcam connected via USB. Since TensorRT optimized models are hardware specific, running the model in TensorRT mode on other hardware would likely cause errors. Use the [model_trt_benchmark.ipynb](model_trt_benchmark.ipynb) notebook to generate alternative Torch-TensorRT models for different hardware. 
