#!/bin/bash
set -e

echo "Installing Node.js and npm for MCP servers..."
apt-get update
apt-get install -y nodejs npm curl
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "Installing GitHub MCP server globally..."
npm install -g @modelcontextprotocol/server-github

echo "MCP server installation complete."
