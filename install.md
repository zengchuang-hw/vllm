# 通过下载预编译包的方式安装

1. 指定 vLLM commit ID（hash from the main branch）
```bash
export VLLM_COMMIT=8fc88d63f1163f119dd740b1666069535f052ff3
```
2. 指定 VLLM 预编译包的下载 URL 
```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```
3. 用可编辑的方式安装
```bash
pip install --editable . -v
```