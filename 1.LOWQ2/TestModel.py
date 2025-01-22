import onnxruntime as ort
import argparse
import numpy as np
from ProcessData import create_arrays

# Parse arguments
parser = argparse.ArgumentParser(description='Train a regression model for the Tagger.')
parser.add_argument('--modelFile', type=str, default="TaggerTrackerTransportation.onnx", help='Path to the ONNX model file')
parser.add_argument('--dataFiles', type=str, nargs='+', help='Path to the data files')
args = parser.parse_args()
modelFile     = args.modelFile
dataFiles     = args.dataFiles

feature_data, target_data = create_arrays(dataFiles)
target_data = np.array(target_data)

# Load the ONNX model
session = ort.InferenceSession(modelFile)

# Run the model on the input data
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
feature_data = np.array(feature_data,dtype=np.float32)
output = session.run([output_name], {input_name: feature_data})
output = np.array(output[0])


polar_angle = np.arctan2(np.sqrt(target_data[:, 0]**2 + target_data[:, 1]**2), target_data[:, 2])

# Multiply columns 0 and 1 by 100
target_data = (np.array(target_data)*np.array([100,100,1]))
output      = (np.array(output)*np.array([100,100,1]))

# Select only events with z > -0.7
target_data_z = target_data[target_data[:, 2] > -0.7]
output_z = output[target_data[:, 2] > -0.7]

# Select only events where the polar angle is < pi-2 mrad
target_data_theta = target_data[polar_angle < np.pi-0.002]
output_theta = output[polar_angle < np.pi-0.002]

# Calculate the rme of the difference between the reference and submitted momenta
def rme(x, y):
    print(x-y)
    return np.sqrt(np.mean((x - y)**2))

rme_momentum = rme(target_data, output)
rme_momentum_z = rme(target_data_z, output_z)
rme_momentum_theta = rme(target_data_theta, output_theta)

print(f"Full data score: {rme_momentum}")
print(f"Energy < 70% beam energy score: {rme_momentum_z}")
print(f"Scattering angle < 2 mrad score: {rme_momentum_theta}")

rme_sum = rme_momentum + rme_momentum_z + rme_momentum_theta
if rme_sum > 1: score = 0
else: score = 1.0 - (np.exp(rme_sum) - 1.0) / (np.exp(1.0) - 1.0)

print(f"Score: {score}")
