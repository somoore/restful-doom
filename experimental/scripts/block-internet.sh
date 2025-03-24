#!/bin/sh
# Script to block outbound internet access while allowing internal communications

# Install iptables if not present
apk add --no-cache iptables

# Allow established connections and related traffic
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow local loopback traffic
iptables -A OUTPUT -o lo -j ACCEPT

# Allow traffic to Docker DNS (container-internal)
iptables -A OUTPUT -d 127.0.0.11 -j ACCEPT

# Allow traffic to internal Docker networks
iptables -A OUTPUT -d 172.16.0.0/12 -j ACCEPT  # Docker networks
iptables -A OUTPUT -d 192.168.0.0/16 -j ACCEPT # Docker networks

# Block all other outbound traffic
iptables -A OUTPUT -j DROP

echo "Outbound internet blocking rules applied successfully"
