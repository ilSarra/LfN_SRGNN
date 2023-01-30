# LfN_SRGNN
Project for the Learning from Networks course of University of Padova.

The oversmoothing problem is a complex issue regarding Graph Neural Networks, since it does not
allow the exploitation of deep architectures, greatly reducing the capabilities of such models. This
problem is quite complex and several techniques have been investigated to solve it. Originally, our
purpose to address oversmoothing was to develop two main techniques involving dimensionality
reduction and ad-hoc regularization. The first intended approach was thought to modify the
aggregation function of GNNs by reducing the number of terms that were summed by using Principal
Component Analysis. The main issue was that we were not able to properly design an implementation
that could guarantee equivariant permutation properties. The complexity of the approach led us to use
a hybrid technique involving both standard aggregation functions and PCA. Moreover, the first
attempts were not satisfactory and we therefore have decided to discard this idea. The code used for
the first trials is available at [7].
For what concerns the regularization term, we have tried to further investigate the problem by
focusing on the embeddings of the network. Results based on similar ideas have already been
presented in the literature [4]. However, while we were trying to better understand how to prevent the
oversmoothing problem by analyzing the hidden representations of a graph neural networks, we came
up with a new simple idea to enhance the capabilities of a network. It is based on the hypothesis that
allowing separate flows of information can allow the network to maintain different or split
representations. This can be achieved by applying two different layers to the same input value to
produce two different outputs. These two outputs can then be combined by means of simple
operations to produce a new value (which can be further transformed or used as the final output of the
network). Note therefore that this architecture is different from a standard sequential model, since we
are operating on the same input with two distinct layers. Further explanations about this technique and
the Split Representation Graph Neural Networks (SRGNN) will be presented below. An additional
improvement has been achieved by introducing a specific regularization term (similar to the one
already designed for the oversmoothing problem) to force the network to differentiate between the
split representations, so that the hidden variables of the two splitted layers are as different as possible
(i.e., avoid the layers to become identical). All the code is available on GitHub [6].
