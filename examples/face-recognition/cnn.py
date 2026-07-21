 
# nvcc -shared -Xcompiler -fPIC -o libextension.so extension.cu -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcudart -lcudnn -lcublas
import time
import sys
import ctypes
import numpy as np
from PIL import Image
import os
import torch 
import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
import torch.nn.functional as F


libextension = ctypes.CDLL("./libextension.so")

test_data = torch.load("test_data.pt")
test_images, test_labels = test_data

class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust based on pooling
        self.fc2 = nn.Linear(128, 40)  # 40 classes (people)

    def forward(self, x):
        x=self.conv1(x)
        x = F.max_pool2d(x, 2)  # Downscale (64 → 32)
        x=self.conv2(x)
        x = F.max_pool2d(x, 2)  # Downscale (32 → 16)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)  

# Load the trained model
model = FaceCNN()  # Ensure model structure is defined
model.load_state_dict(torch.load("face_cnn.pth"))
model.eval()

# Extract weights and biases
weights_conv1 = model.conv1.weight.detach().numpy().astype(np.float32) 
bias_conv1 = model.conv1.bias.detach().numpy().astype(np.float32) 
weights_conv2 = model.conv2.weight.detach().numpy().astype(np.float32) 
bias_conv2 = model.conv2.bias.detach().numpy().astype(np.float32) 
weights_fc1 = model.fc1.weight.detach().numpy().astype(np.float32) 
bias_fc1 = model.fc1.bias.detach().numpy().astype(np.float32)  
weights_fc2 = model.fc2.weight.detach().numpy().astype(np.float32)  
bias_fc2 = model.fc2.bias.detach().numpy().astype(np.float32) 

weights_dict = {
    "weights_conv1": weights_conv1,
    "bias_conv1": bias_conv1,
    "weights_conv2": weights_conv2,
    "bias_conv2": bias_conv2,
    "weights_fc1": weights_fc1,
    "bias_fc1": bias_fc1,
    "weights_fc2": weights_fc2,
    "bias_fc2": bias_fc2
}

# class CNN():
#     def __init__(self):
#         super(CNN, self).__init__()

#         self.weights_dict=weights_dict

#     def forward(self, x, output):
#         x_in=x.reshape(64,64).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         weights1=self.weights_dict["weights_conv1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         bias1=self.weights_dict["bias_conv1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         weights2=self.weights_dict["weights_conv2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         bias2=self.weights_dict["bias_conv2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         weightsfc1=self.weights_dict["weights_fc1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         biasfc1=self.weights_dict["bias_fc1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         weightsfc2=self.weights_dict["weights_fc2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         biasfc2=self.weights_dict["bias_fc2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         #x = conv(x,self.weights, self.bias)
#         #x = subsample(x,self.subsample_weights, self.subsample_bias)
#         #x = fully_connect(x,self.weights_fc,self.bias_fc)
#         libextension.forward_pass(x_in, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), weights1, bias1, weights2, bias2,weightsfc1, biasfc1, weightsfc2, biasfc2)
#         return x

# # Instantiate the model
# model = CNN()

# def test_model():
#     libextension.forward_pass.argtypes = [
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float), 
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float), 
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ]
#     # model.eval()  # Set the model to evaluation mode
#     correct = 0
#     total = 0
#     directory="images/"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     total=10
#     for i in range(total):
#         inputs=test_images[i].numpy().squeeze()
#         labels=test_labels[i].numpy().squeeze()
#         image = Image.fromarray((inputs*255).astype(np.uint8))
#         image.save(f"{directory}test_image_{i}.png",)
#         # image.show()

#         outputs=np.zeros((40,), dtype=np.float32)
#         model.forward(inputs,outputs)
#         predicted = np.argmax(outputs)
#         print(predicted,labels)
#         correct += (predicted == labels).sum().item()

#     print(f"Test Accuracy: {100 * correct / total}%")

# Evaluate the model

def configure_extension():
    weights1 = weights_dict["weights_conv1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias1 = weights_dict["bias_conv1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weights2 = weights_dict["weights_conv2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias2 = weights_dict["bias_conv2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weightsfc1 = weights_dict["weights_fc1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    biasfc1 = weights_dict["bias_fc1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weightsfc2 = weights_dict["weights_fc2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    biasfc2 = weights_dict["bias_fc2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    libextension.create_model.restype = ctypes.c_void_p
    libextension.create_model.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    libextension.forward_pass.restype = None
    libextension.forward_pass.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    model_handle = libextension.create_model(
        weights1, bias1,
        weights2, bias2,
        weightsfc1, biasfc1,
        weightsfc2, biasfc2,
    )

    return model_handle


def run_dataset(model_handle, save_images=False):
    correct = 0
    total = int(os.environ.get("STEADY_IMAGES", "10"))

    if save_images:
        directory = "images/"
        os.makedirs(directory, exist_ok=True)

    for i in range(total):
        inputs = test_images[i].numpy().squeeze()
        label = int(np.asarray(test_labels[i]).squeeze())

        if save_images:
            image = Image.fromarray((inputs * 255).astype(np.uint8))
            image.save(f"images/test_image_{i}.png")

        x_in = inputs.reshape(64, 64).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        outputs = np.zeros((40,), dtype=np.float32)

        libextension.forward_pass(
            model_handle,
            x_in,
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        predicted = int(np.argmax(outputs))
        correct += int(predicted == label)

    return correct, total


if __name__ == "__main__":
    warmups = int(os.environ.get("STEADY_WARMUPS", "5"))
    measured_iterations = int(os.environ.get("STEADY_ITERS", "100"))
    save_images = os.environ.get("SAVE_IMAGES", "0") == "1"

    model_handle = configure_extension()

    for i in range(warmups):
        run_dataset(model_handle, save_images=(save_images and i == 0))
        print(f"Warmup Completed: {i + 1}/{warmups}", flush=True)

    total_correct = 0
    total_images = 0

    measured_start = time.perf_counter()

    for i in range(measured_iterations):
        correct, count = run_dataset(model_handle, save_images=False)
        total_correct += correct
        total_images += count

        if (i + 1) % 5 == 0 or (i + 1) == measured_iterations:
            print(f"Measured Iteration Completed: {i + 1}/{measured_iterations}", flush=True)

    measured_end = time.perf_counter()
    measured_time = measured_end - measured_start

    accuracy = 100.0 * total_correct / total_images if total_images else 0.0
    avg_iter = measured_time / measured_iterations if measured_iterations else 0.0
    avg_image = measured_time / total_images if total_images else 0.0

    print(f"Warmup Iterations: {warmups}", flush=True)
    print(f"Measured Iterations: {measured_iterations}", flush=True)
    print(f"Images Per Iteration: {int(os.environ.get('STEADY_IMAGES', '10'))}", flush=True)
    print(f"Total Measured Images: {total_images}", flush=True)
    print(f"Test Accuracy: {accuracy}%", flush=True)
    print(f"Execution Time: {measured_time:.6f} seconds", flush=True)
    print(f"Average Iteration Time: {avg_iter:.6f} seconds", flush=True)
    print(f"Average Time: {avg_image:.6f} seconds", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()

    # Avoid Python/C++ shared-library teardown affecting the benchmark result.
    os._exit(0)