网络代理与下载加速指南

目的
- 配置 git 与 Hugging Face 访问代理，加速或解决下载超时/连不上的问题。
- 提供训练时一键注入代理环境的方式。

代理设置
- Git 级别（可选）：
  - `git config --global http.proxy http://100.64.117.161:3128`
  - `git config --global https.proxy http://100.64.117.161:3128`
  - `git config --global http.https://github.com.proxy http://100.64.117.161:3128`

- 环境变量（推荐在训练/下载前导出）：
  - `export https_proxy="http://100.64.117.161:3128"`
  - `export http_proxy="http://100.64.117.161:3128"`

注意事项
- 不要同时开启代理与设置 Hugging Face 镜像 (`HF_ENDPOINT`)。两者同时用会明显变慢。
- 如果需要改用镜像（或临时关闭代理），先执行：
  - `unset https_proxy; unset http_proxy`

缓存目录
- 默认将 Hugging Face 缓存写到项目内：`hf_cache/`，避免系统默认目录空间不足。
- 相关环境变量：
  - `HF_HOME=$(pwd)/hf_cache`
  - `TRANSFORMERS_CACHE=$(pwd)/hf_cache/hub`（Transformers v5 起建议仅使用 `HF_HOME`）

一键训练（带代理）
- 推荐使用脚本：`scripts/train_with_proxy.sh`。示例：
  - `bash scripts/train_with_proxy.sh --config configs/heart_mambaformer_small.yaml --gpus 0 --output runs`
- 日志输出位置：`logs/train_*.log`

