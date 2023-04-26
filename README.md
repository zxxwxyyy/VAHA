# VAHA - Visual Artwork for Human Affections
 
[NYU Tandon, Deep Learning for Media Final Project]

[Liqian Zhang](), [Yunfeng Qi](), [Joanne Tang]()

[[`Demo`](https://colab.research.google.com/drive/1sGToDW9JF8Q5iSagNdZ5_ornuEncvPl5?usp=sharing)]
[[`Presentation`](https://docs.google.com/presentation/d/1ZYf7WW_uSSE5V4EUYXyrYrDDmwPbSk9NadjnbihCkDs/edit?usp=sharing)]

![t2i](assets/final_results.png)


## **Introduction**

The **Visual Artwork for Human Affections(VAHA)** is an innovative model designed to explore the intersection of creativity, human emotions, and artificial intelligence to revolutionize the way we understand and express our emotions. Our goal is to provide a deeper understanding of the complexity of human emotions by automating the generation of abstract visual artwork using emotion recognition and image generation techniques.

The **VAHA** model integrates two established models: the VGG16 and the GAN. The VGG16, a Convolutional Neural Network (CNN) model, is specifically adapted for facial recognition in our system. The GAN (Generative Adversarial Network) model is responsible for generating the visual art images by integrating the classifications of human emotions obtained from the VGG16 model. The **VAHA model can produce captivating artistic representations of various emotional states, incorporating a diverse range of associated artistic styles.**

Check our demo on how to use. 

## **Dataset**

We used [[`FER-2013 Dataset`](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?select=train)] to trained our emotion detector. And we used [[`Wiki-art Dataset`](https://www.kaggle.com/datasets/steubk/wikiart)] with [[`Artemis`](https://www.artemisdataset.org/)] annotated labels to trained our cGAN model. 

## **Models** 

### Emotion detector 

Through our [[model experiment](https://github.com/zxxwxyyy/VAHA/blob/main/Notebooks/VAHA_emotion_detect_model_experiment.ipynb)], we notice [[`vgg16`](https://keras.io/api/applications/vgg/)] offers the best performance. So we set all layers to trainable, and trained it with FER-2013 dataset for 20 epochs to adpat features on our specific needs. 

![d2i](assets/model_compare.png)
<!-- ![d2i](assets/vgg_16.jpg) -->

### Conditional Generative Adversarial Network (cGAN)

Our VAHA model is trained on [[`Wiki-art Dataset`](https://www.kaggle.com/datasets/steubk/wikiart)] with [[`Artemis`](https://www.artemisdataset.org/)] emotion labels. Our generator takes random noise and a class label as input, and generates images conditioned on the class label. The label embedding and concatenation layers allows the generator to incorporate the class label information into generated image, making the output class specificed. 

![d2i](assets/cGAN_model.png)

### Artemis pre-trained model

We input all generated images through Artemis pre-trained model, and take the 3 images that has the highest prediction probabilities as outputs. 

### Real-ESRGAN denoising model 

Because of the limited computing resources, the output images of our cGAN is in 64x64x3. So we take our outputs images in Real-ESRGAN denoising model for further denoising. 

## **Citation** 

Big thanks to all resources we've been used: 

Artemis:
```
     @article{achlioptas2021artemis,
                title={ArtEmis: Affective Language for Visual Art},
                author={Achlioptas, Panos and Ovsjanikov, Maks and Haydarov,
                        Kilichbek and Elhoseiny, Mohamed and Guibas, Leonidas},
                journal = {CoRR},
                volume = {abs/2101.07396},
                year={2021}}
```

Real-ESRGAN: 
```
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```

