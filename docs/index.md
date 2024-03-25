# Welcome to MkDocs

The MADNESS Deblender is being developed with the LSST DESC Collaboration to deblended galaxies from a blended source using machine learning. The algorithm obtains the maximum a posterior solution to the inverse problem of deblending using Variational Auto Encoders (VAEs) and Normalizing Flows (NFs).

This python package allows easy modification of the neural network architecture and can easily be retrained for different Surveys.

## Installation

For testing the deblender, the package can directly be installed from GitHub
```bash
pip install git+https://github.com/b-biswas/MADNESS.git
```
A release is being planned soon.

## Getting Started

MADNESS Deblender is built on Tensorflow and Tensorflow Probability to allow seamless useage of GPUs for training and deblending.

### Initialization
The model present in the final paper is available at TODO and can be used to reproduce the results of the paper.

```
from maddeb.Deblender import Deblend
from galcheat import Survey

# initialize
galcheat.get_survey(maddeb_config["LSST"])
deb = Deblend(weights_path=weights_path, survey=survey)
```

### Run the Deblender
MADNESS is designed to run the deblender on multiple fields in parallel to improve speed. The necessary inputs to run the deblender inlude:<br />
- a batch of fields (unnormalized) <br />
- list of detected positions in each field<br />
- list of the number of components in each field<br />
For stability of the code, the list of detected positions for each field must be the same (appended with zeroes if required).

Calling the instance of maddeb.Deblend runs the gradient descent algorithm.
```
deb(
    blended_fields,
    detected_positions,
    num_components,
)
```
Further parameters can be passed to select the linear normalizing coefficient, learning rate, optimizer, and stopping criterion for the gradient descent.
See the Deblending tutorial for more information.
### Accessing Results
Although the solutions are obtained in the latent space, the corresponding image can direcctly be accessed though the instance of maddeb.Deblend.
```
field0_results = deb.components[0] # All deblended galaxies in the first field
field1_results = deb.components[1] # All deblended galaxies in the second field
```
