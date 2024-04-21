[English](./README.md) | **[中文](./README_tw.md)**

# DocClassifier

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocClassifier/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocClassifier?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## 介紹

<div align="center">
    <img src="./docs/title.jpg" width="800">
</div>

DocClassifier是一個基於Metric Learning技術的文件圖像分類系統，受到傳統分類器面臨的挑戰啟發，解決了文件類型快速增加和難以定義的問題。它採用了PartialFC特徵學習架構，集成了CosFace和ArcFace等技術，使模型能夠在不需要大量預定義類別的情況下進行精確分類。通過擴展數據集和引入ImageNet-1K和CLIP模型，我們提高了性能並增加了模型的適應性和可擴展性。模型使用PyTorch進行訓練，在ONNXRuntime上進行推理，並支持將模型轉換為ONNX格式以便在不同平台上部署。在測試中，我們的模型達到了超過90%的準確率，並具有快速推理速度和快速添加新文件類型的能力，從而滿足了大多數應用場景的需求。

## 快速開始

套件安裝和使用的方式，請參閱 [**DocClassifier Documents**](https://docsaid.org/docclassifier/intro/)。

---

## 引用

我們感謝所有走在前面的人，他們的工作對我們的研究有莫大的幫助。

如果您認為我們的工作對您有幫助，請引用以下相關論文：

```bibtex
@misc{lin2024docclassifier,
  author = {Kun-Hsiang Lin, Ze Yuan},
  title = {DocClassifier},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.10538356},
  howpublished = {\url{https://github.com/DocsaidLab/DocClassifier}}
}

@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}

@misc{gpiosenka_cards_2023,
  author = {Gerry},
  title = {{Cards Image Dataset for Classification}},
  year = {2023},
  howpublished = {\url{https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification?resource=download}},
  note = {Accessed: 2024-01-19},
  license = {CC0: Public Domain}
}

@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}

@inproceedings{wang2018cosface,
  title={Cosface: Large margin cosine loss for deep face recognition},
  author={Wang, Hao and Wang, Yitong and Zhou, Zheng and Ji, Xing and Gong, Dihong and Zhou, Jingchao and Li, Zhifeng and Liu, Wei},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5265--5274},
  year={2018}
}
```
