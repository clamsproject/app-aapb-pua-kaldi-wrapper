FROM brandeisllc/aapb-pua-kaldi:v2
# which has python3.7
# may not want to do apt-get update if there are dependencies of
# the Kaldi image that rely on older versions of apt packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-setuptools

COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt

CMD ["python3", "app.py", "--production"]
