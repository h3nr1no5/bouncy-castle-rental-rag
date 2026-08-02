FROM grafana/grafana:10.4.3

COPY grafana/provisioning /etc/grafana/provisioning
COPY grafana/dashboard.json /var/lib/grafana/dashboards/dashboard.json
COPY grafana/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
