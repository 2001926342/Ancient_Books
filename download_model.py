from modelscope import snapshot_download
import os
import streamlit as st

# 下载古籍模型
def download_ancient_books_model():
    model_dir = snapshot_download('CFYuan/Ancient_Books', cache_dir='/group_share/Ancient_Books/model')
    return model_dir
# 下载lint4量化古籍模型
def download_ancient_books_model_int4():
    model_dir = snapshot_download('CFYuan/Ancient_Books_int4B', cache_dir='/group_share/Ancient_Books/model')
    return model_dir

if __name__ == '__main__':
    下载所需模型
    download_ancient_books_model()
    download_ancient_books_model_int4()
