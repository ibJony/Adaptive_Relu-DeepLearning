### ABSTRACT: Proposing a Adaptive Relu (AdaReLU) 

Rectified linear units (RELUs) are the most popular activation function used in neural networks because it can alleviate the gradient vanishing problem and improve
the convergence rate. Variants of ReLUs such as Leaky ReLU and PReLU have also achieved improved performance in many deep learning tasks. In this
project, we study adaptive Relu (AdaReLU) that parameterizes the activation threshold used in ReLUs. In convolutional neural networks, AdaReLU can be
combined with parameter-sharing schemes and a partial replacement strategy to construct an adaptive convolutional ReLU (ConvReLU) block. ConvReLU is
claimed to solve the dying ReLU problem and consistently perform better than ReLU, LeakyReLU, and PReLU. However, the experiments and analysis conducted
in the paper are not convincing enough to show that the current ConvReLU model is better than LeakyReLU and PReLU that have already been proven useful.
In this project, we will implement the proposed methods, provide more detailed analysis, and also experiment on different implementations.
