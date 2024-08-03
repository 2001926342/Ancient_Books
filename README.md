# Ancient_Books
<div align="center">
  <img src="./assets/logo.png" width="600"/>
  <!-- <a href="https://github.com/Nobody-ML/SoulStar/tree/main/">
    <img src="assets/logo.png" alt="Logo" width="600">  </a> -->
  <h3 align="center">Ancient_Books - 古籍解读大模型</h3>
</div>

开源不易，如果本项目帮到大家，可以右上角帮我点个 star~ ⭐

您的 star ⭐是我们最大的鼓励，欢迎 Star⭐、PR 和 Issue。

 ## 📖 目录
- [Ancient_Books - 古籍解读大模型](#Ancient_Books---古籍解读大模型)
  - [📖 目录](#-目录)
  - [🔄 架构图](#-架构图)
  - [📝 简介](#-简介)
  - [🛠️ 使用方法](#️-使用方法)
    - [1.下载模型](#1-下载模型)
    - [2.环境部署](#2-环境部署)
    - [3.本地部署](#3-本地部署)
  - [🧾 数据来源](#-数据来源)
  - [🧑‍💻 微调指南](#-微调指南)
  - [🧑‍💻 RAG指南](#-RAG指南)
  - [📚 应用体验](#-应用体验)
  - [🎖️ 致谢](#️-致谢)
 
## 🔄 架构图


## 📝 简介

甘肃政法大学人工智能学院推出的古籍解读大模型是一款辅助学习的工具，专为帮助用户理解和欣赏中国古代文学和文化而设计。它具备古诗赏析、文言文翻译、成语解释、《论语》注释以及《百家姓》解读等功能，使用户能够深入领会古代诗词、文献、成语典故和姓氏文化的精髓，是学术研究者、学生以及所有对中国古代文化感兴趣者的理想助手。

## 🛠️ 使用方法

### 快速开始

1.下载模型


参考 [模型的下载]( https://modelscope.cn/models/CFYuan/Ancient_Books) 。

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
model_dir = snapshot_download('CFYuan/Ancient_Books')
```


或者参考文件 download_model.py ，支持7B模型与7B int4 量化后的模型

```python
python  download_model.py
python  download_hf.py
```

2.环境部署

```bash
git clone https://github.com/2001926342/Ancient_Books

pip install requirements.txt
```
3.本地部署

```python
streamlit run web.py --server.port 7860
```


## 🧾 数据来源

一下是项目目前使用到的开源数据集，还使用爬虫技术获取我们所需数据集：

文言文：https://huggingface.co/datasets/RUCAIBox/Erya-dataset/tree/main

古诗：https://github.com/chinese-poetry/chinese-poetry

文言文（古文）- 现代文平行语料：https://github.com/NiuTrans/Classical-Modern


## 🧑‍💻 微调指南



## 🧑‍💻 RAG指南



## 💕 致谢

### 项目成员

- 陈辅元-项目负责人 （甘肃政法大学 Datawhale鲸英助教 负责模型微调训练，数据收集，RAG内容整理）
- 张世斌-项目负责人 （甘肃政法大学）
- 柴承清 （甘肃政法大学）
- 李智江 （甘肃政法大学）
- 符银霞 （甘肃政法大学）

### 特别鸣谢
<div align="center">
 
***感谢上海人工智能实验室组织的 书生·浦语实战营 学习活动~***

***感谢 OpenXLab 对项目部署的算力支持~***

***感谢 浦语小助手 对项目的支持~***

***感谢上海人工智能实验室推出的书生·浦语大模型实战营，为我们的项目提供宝贵的技术指导和强大的算力支持！***
