import awkward as ak
import numpy as np
import uproot

reference = uproot.open({"targets_lowq2.edm4eic.root": "events"})
submit    = uproot.open({"submit_lowq2.edm4eic.root": "events"})

# Full data set
reference_momentum = reference["_TaggerTrackerTargetTensor_floatData"].array()
submitted_momentum = submit["_TaggerTrackerPredictionTensor_floatData"].array()

# Select only events with a single track
num_tracks = submit["_TaggerTrackerPredictionTensor_shape"].array()[:,0]
reference_momentum = reference_momentum[num_tracks == 1]
submitted_momentum = submitted_momentum[num_tracks == 1]

polar_angle = np.arctan2(np.sqrt(reference_momentum[:, 0]**2 + reference_momentum[:, 1]**2), reference_momentum[:, 2])

# Multiply columns 0 and 1 by 100
reference_momentum = (np.array(reference_momentum)*np.array([100,100,1]))
submitted_momentum = (np.array(submitted_momentum)*np.array([100,100,1]))

# Select only events with z > -0.7
reference_momentum_z = reference_momentum[reference_momentum[:, 2] > -0.7]
submitted_momentum_z = submitted_momentum[reference_momentum[:, 2] > -0.7]

# Select only events where the polar angle is < pi-2 mrad
reference_momentum_theta = reference_momentum[polar_angle < np.pi-0.002]
submitted_momentum_theta = submitted_momentum[polar_angle < np.pi-0.002]

# Calculate the rme of the difference between the reference and submitted momenta
def rme(x, y):
    print(x-y)
    return np.sqrt(np.mean((x - y)**2))

rme_momentum = rme(reference_momentum, submitted_momentum)
rme_momentum_z = rme(reference_momentum_z, submitted_momentum_z)
rme_momentum_theta = rme(reference_momentum_theta, submitted_momentum_theta)

print(f"Full data score: {rme_momentum}")
print(f"Energy < 70% beam energy score: {rme_momentum_z}")
print(f"Scattering angle < 2 mrad score: {rme_momentum_theta}")

rme_sum = rme_momentum + rme_momentum_z + rme_momentum_theta
if rme_sum > 1: score = 0
else: score = 1.0 - (np.exp(rme_sum) - 1.0) / (np.exp(1.0) - 1.0)

print(f"Score: {score}")
