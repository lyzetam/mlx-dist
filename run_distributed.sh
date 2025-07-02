#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting MLX Distributed Inference${NC}"
echo "================================================"

# Check if hostfile exists
if [ ! -f "hostfile" ]; then
    echo -e "${RED}❌ Error: hostfile not found!${NC}"
    exit 1
fi

# Check if Python script exists
if [ ! -f "distributed_inference.py" ]; then
    echo -e "${RED}❌ Error: distributed_inference.py not found!${NC}"
    exit 1
fi

# Show configuration
echo -e "${YELLOW}📋 Configuration:${NC}"
echo "Hostfile contents:"
cat hostfile
echo ""

# Test SSH connectivity first
echo -e "${YELLOW}🔌 Testing SSH connections...${NC}"
ssh -o BatchMode=yes -o ConnectTimeout=5 10.85.35.29 'echo "✅ Mac mini 1 connected"'
ssh -o BatchMode=yes -o ConnectTimeout=5 10.85.35.205 'echo "✅ Mac mini 2 connected"'

# Run distributed inference
echo -e "\n${GREEN}🏃 Running distributed inference...${NC}"
echo "Command: mpirun -np 3 --hostfile hostfile python distributed_inference.py"
echo "================================================"

# Use full Python path to ensure correct environment
PYTHON_PATH="/opt/homebrew/bin/python3.11"

# Run with error checking
if mpirun -np 3 --hostfile hostfile \
    --mca btl_tcp_if_include en0 \
    --mca btl_tcp_links 4 \
    $PYTHON_PATH distributed_inference.py; then
    echo -e "\n${GREEN}✅ Distributed inference completed successfully!${NC}"
else
    echo -e "\n${RED}❌ Distributed inference failed!${NC}"
    echo "Check the error messages above for details."
    exit 1
fi