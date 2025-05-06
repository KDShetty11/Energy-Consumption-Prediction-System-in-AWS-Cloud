FROM ubuntu:20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-numpy \
    python3-pandas \
    openjdk-11-jdk \
    wget \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz && \
    tar xvf spark-3.5.5-bin-hadoop3.tgz -C /opt && \
    rm spark-3.5.5-bin-hadoop3.tgz && \
    ln -fs /opt/spark-3.5.5-bin-hadoop3 /opt/spark

# Set Spark environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Configure Spark logging
RUN cp $SPARK_HOME/conf/log4j2.properties.template $SPARK_HOME/conf/log4j2.properties && \
    sed -i 's/rootLogger.level = info/rootLogger.level = ERROR/g' $SPARK_HOME/conf/log4j2.properties

# Copy application files
COPY predict.py /app/
COPY regressionmod /app/regressionmod

# Set entrypoint
ENTRYPOINT ["spark-submit", "predict.py"]

# Default command - will be overridden if args are provided to docker run
CMD [""]