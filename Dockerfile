FROM python:3.8
LABEL maintainer="job.tanaj"

ENV USER=root

# install relevant packages
RUN apt-get update -yqq
RUN apt-get upgrade -yqq
RUN apt-get install -yqq nano
RUN apt-get install -yqq curl
RUN apt-get install -yqq wget
RUN apt-get install -yqq vim
RUN apt-get install -yqq tree

# make the new directory in the container to host the algorithm
RUN mkdir -p /

# copy the algorithm to the destination folder
COPY . /

# copy the dependencies file to the working directory
COPY requirements.txt /

# install python dependencies
RUN pip install -r requirements.txt

# Expose port 5000 inside the container
EXPOSE 5000

# command to run ca3_flask_main.py with python
CMD [ "python" , "/ca3_flask_main.py" ]
