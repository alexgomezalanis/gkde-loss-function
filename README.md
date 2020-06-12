# A Gaussian Kernel Density Estimation Loss Function
Paper: A. Gomez-Alanis, J. A. Gonzalez-Lopez and A. M. Peinado, "A Kernel Density Estimation Based Loss Function and Its Application to ASV-Spoofing Detection," in IEEE Access, doi: 10.1109/ACCESS.2020.3000641.

## Abstract

Biometric systems are exposed to spoofing attacks which may compromise their security, and voice biometrics, also known as automatic speaker verification (ASV), is no exception. Replay, synthesis and voice conversion attacks cause false acceptances that can be detected by anti-spoofing systems. Recently, deep neural networks (DNNs) which extract embedding vectors have shown superior performance than conventional systems in both ASV and anti-spoofing tasks. In this work, we develop a new concept of loss function for training DNNs which is based on kernel density estimation (KDE) techniques. The proposed loss functions estimate the probability density function (pdf) of every training class in each mini-batch, and compute a log likelihood matrix between the embedding vectors and pdfs of all training classes within the mini-batch in order to obtain the KDE-based loss. To evaluate our proposal for spoofing detection, experiments were carried out on the recent ASVspoof 2019 corpus, including both logical and physical access scenarios. The experimental results show that training a DNN based anti-spoofing system with our proposed loss functions clearly outperforms the performance of the same system being trained with other well-known loss functions. Moreover, the results also show that the proposed loss functions are effective for different types of neural network architectures.

## Log Likelihood Matrix

![Alt text](/images/log_likelihood_matrix.png?raw=true "System overview for computing the log likelihood matrix of a mini-batch of $N \times M$ utterances.")

System overview for computing the log likelihood matrix of a mini-batch of N x M utterances.