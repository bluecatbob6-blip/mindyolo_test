FROM 10.10.102.110:443/chief-ai-centerplatform/yolov5_train:v0.2.2

WORKDIR /root/mindyolo_train

COPY ./requirements.txt /root/mindyolo_train/requirements.txt

RUN pip install -r requirements.txt --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . /root/mindyolo_train
