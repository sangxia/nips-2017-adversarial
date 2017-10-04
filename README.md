# Adversarial Attacks and Defenses of Image Classifiers, NIPS 2017 competition track

> *From [NIPS 2017 competition track website](https://nips.cc/Conferences/2017/CompetitionTrack)*
> > Most existing machine learning classifiers are highly vulnerable to adversarial examples. 
> > An adversarial example is a sample of input data which has been modified very slightly in 
> > a way that is intended to cause a machine learning classifier to misclassify it. In many 
> > cases, these modifications can be so subtle that a human observer does not even notice the 
> > modification at all, yet the classifier still makes a mistake. Adversarial examples pose 
> > security concerns because they could be used to perform an attack on machine learning systems, 
> > even if the adversary has no access to the underlying model.
> > 
> > To accelerate research on adversarial examples and robustness of machine learning classifiers 
> > we organize a challenge that encourages researchers to develop new methods to generate 
> > adversarial examples as well as to develop new ways to defend against them. As a part of the 
> > challenge participants are invited to develop methods to craft adversarial examples as well 
> > as models which are robust to adversarial examples.

There are three sub-competitions

* [Non-targeted Attack](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack)
* [Targeted Attack](https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack)
* [Defense](https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack)

Dev dataset and toolkit available 
[here](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition).

## Submission source code

My submission to the three sub-competitions are in the `submissions` directory. 

Files containing pretrained weights need to be placed in the working directory. All `.h5` weights
are from [Keras](https://keras.io/). All `.ckpt` weights are from 
[TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim).

In the actual submission, the final graphs are frozen, serialized and saved to a `.pb` file for 
fast loading. This is done with the `prepare_models.py` script.

Note that the submission code as is has some known bugs as commented in the code.

