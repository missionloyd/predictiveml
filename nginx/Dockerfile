FROM nginx:latest

RUN apt-get update && apt-get install -y python3-certbot-nginx

RUN rm /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/
RUN rm /etc/nginx/conf.d/default.conf
COPY project.conf /etc/nginx/conf.d/


EXPOSE 80
EXPOSE 443