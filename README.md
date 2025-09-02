# ✨Getting Started

## Installation

You can install LUFFY dependencies by running the following commands:
```bash
conda create -n rlamg python=3.10
conda activate rlamg
cd rlamg
cd verl
# Make sure you have activated verl conda env
# If you need to run with megatron
# bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
cd ..
pip install -r requirements.txt
```

## proxy setting (optional: for wandb)
Install clash for linux by running:
```bash
wget https://github.com/wnlen/clash-for-linux/raw/refs/heads/master/bin/clash-linux-amd64
mv clash-linux-amd64 clash
chmod +x clash
```
Then get your clash config file and put it in ~/.config/clash.

Run clash in the background:
```bash
# run clash
clash -d ~/.config/clash #-d: set clash directory

chmod +x switch_proxy.sh # make switch_proxy.sh executable
./switch_proxy.sh "香港—E3" # switch to proxy node

# nohup clash
nohup clash -d ~/.config/clash > ~/.config/clash/clash.log 2>&1 &

# set proxy
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="socks5://127.0.0.1:7890"

# close clash
ps aux | grep clash # find PID of clash process
kill -9 <PID> # kill clash process
```
# RLAMG
