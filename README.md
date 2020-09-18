# Anime Face generation: Generates faces of Anime characters using [Generative Adverserial Networks (GANs)](https://en.wikipedia.org/wiki/Generative_adversarial_network)


This Anime Face Generator will be trained based on DCGAN, visualize with [visdom](https://github.com/facebookresearch/visdom).

## Data
The data used for generator training is collected by [DCGAN_IMG_COLLECTOR.py](https://github.com/lipilian/NewProject_dontKnowNameYet/blob/master/DCGAN_IMG_COLLECTOR.py), which collected raw anime image from [konachan](https://konachan.net/).

For face detection, [DCGAN_FACE_DETECTOR.py](https://github.com/lipilian/NewProject_dontKnowNameYet/blob/master/DCGAN_FACE_DETECTOR.py) was used for face feature detection, which thanks to the github User [bchao1](https://github.com/bchao1/Anime-Face-Dataset/blob/master/src/scrape.py)


## Todo list

- [x] DCGAN
- [x] connect to visdom
- [x] checkpoints saving
- [ ] Add Resnet structure to improve net
- [ ] Try condintional Gan net
  
## Some resources

- [self attention GAN in PyTorch](https://github.com/heykeetae/Self-Attention-GAN)
- [gan hacks and tips](https://github.com/soumith/ganhacks)
- [batch norm](https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0)
- [CVPR 2018 Tutorial on GANs](https://sites.google.com/view/cvpr2018tutorialongans/)
- [auxiliary classifier GAN ac gan from scratch with keras](https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/)

## Results

Progress during training (epochs 0 to 400):

![progress_gif](assets/DCGAN.gif)

64x64 Generated samples after 20 epochs. Training time: ~5 hours on x2 GTX 1080-Ti GPUs


![96x96](assets/generated_sample_96x96(epoch45).png)

## [Tensorboard](https://www.tensorflow.org/tensorboard)

A lot of work was put into the checkpointing system and logging, as well as writing to tensorboard.

Histograms of each layer and its gradient



## [Certificate to *Udacity Connect Deep Learning NanoDegree*](https://graduation.udacity.com/confirm/CTG23EKQ)


