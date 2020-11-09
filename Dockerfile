FROM hipstas/kaldi-pop-up-archive:v1

LABEL maintainer="Angus L'Herrou <piraka@brandeis.edu>"

# hipstas/kaldi-pop-up-archive:v1 uses Ubuntu 16.10 Yakkety, which is dead, so no apt repositories.
# Have to tell apt to use Ubuntu 18.04 Bionic's apt repositories, since that's the oldest LTS with
# Python 3.6. This is terrible!
RUN cp /etc/apt/sources.list /etc/apt/sources.list.old && \
    sed -i -e s/yakkety/bionic/g /etc/apt/sources.list

# may not want to do apt-get update if there are dependencies of
# the Kaldi image that rely on older versions of apt packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-setuptools

COPY ./ ./app
WORKDIR ./app
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["app.py"]
