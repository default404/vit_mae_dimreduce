# ViT MAE for Image Latent Space Embedding

This implementation uses the Transformer architecute to embed images into a meaningful, low dimensional latent space forto use the reduced number of input features to train quantum machine learning (QML) models.

The AE model is an implementation of the Vision Transformer (ViT) from [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) by He et. al. for self-supervised pretraining in the _Masked _Image_ _Modeling_ setting.

Code example is an implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) and inspired by [keras.io example](https://github.com/keras-team/keras-io/blob/master/examples/vision/masked_image_modeling.py) .

Example Notebooks provided for [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [MNIST](http://yann.lecun.com/exdb/mnist/) .
