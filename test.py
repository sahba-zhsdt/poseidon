import torch
print(torch.cuda.is_available())  # Should return True if CUDA is set up correctly
print(torch.cuda.get_device_name(0))  # Should print the GPU name if available
print("Number of available GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{i}")
    print(f"Testing GPU {i}: ", torch.cuda.get_device_name(device))

# print(torch.backends.nccl.is_available())  # Should print True if NCCL is available
print(torch.distributed.is_nccl_available())  # Another check for NCCL availability
print(torch.__version__)
print(torch.version.cuda)


