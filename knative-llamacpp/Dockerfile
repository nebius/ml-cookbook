FROM ghcr.io/ggml-org/llama.cpp:server-cuda
LABEL authors="brian"
RUN apt update
RUN apt install unzip
RUN apt clean
RUN apt autoremove
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]