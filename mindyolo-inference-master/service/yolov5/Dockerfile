# FROM python:3.8-vim
FROM 10.10.102.110:443/chief-ai-centerplatform/yolov5-inference:v0.2.0

COPY . /root/yolov5_train

WORKDIR /root/yolov5_train

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple