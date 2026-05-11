FROM 10.10.102.110/chief-ai-centerplatform/yolov5-inference:v1.0.0

COPY . /mindyolo_inference

WORKDIR /mindyolo_inference

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir \
    && sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list \
    && apt update -y \
    && apt install -y libgl1-mesa-glx --no-install-recommends
