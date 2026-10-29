# 模型准备（这一步比装依赖麻烦，先说清为什么）

## 为什么不用 `ollama pull minicpm-v`

在本次实施环境里实测遇到两个问题，都不是配置错误：

| 问题 | 现象 | 原因 |
|------|------|------|
| 卡在 `pulling manifest` 不动 | 等十几分钟一直在转圈 | ollama 服务进程没有写 `~/.ollama/models/manifests` 的权限（服务日志明确报 `operation not permitted`）。用 `brew services` 或直接在终端前台启动服务没有这个问题 |
| 下载速度 271 KB/s | 5.5GB 需要约 5.6 小时 | 走 `registry.ollama.ai` 的 CDN，当前网络下就是这个速度 |

## 推荐路径：本地 GGUF 导入（快，且用面壁官方模型）

走 HuggingFace 下载面壁官方的 GGUF，再用 `ollama create` 导入，绕过 registry。
MiniCPM5-2B 量化后约 1.4GB，几分钟能下完。

```bash
# 1. 下载到项目 models/ 目录（该目录已在 .gitignore 中，不会误入版本库）
mkdir -p models
curl -L -o models/MiniCPM5-2B-Q4_K_M.gguf \
  "https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/resolve/main/MiniCPM5-2B-Q4_K_M.gguf"

# 2. 导入（在项目根目录执行，Modelfile.minicpm5 已备好）
ollama create minicpm5-2b -f Modelfile.minicpm5

# 3. 验证
ollama run minicpm5-2b "你好"
```

> **实测结果**：两个方案最终都跑通了。`ollama pull minicpm-v` 在服务权限修好、网络换成家庭宽带后正常拉取（5.5GB），
> 并且**最终 POC 测试用的是 `minicpm-v`**——同样语料下它的条款复述完整度明显好于 2B，
> `minicpm5-2b` 作为"低配办公本"方案保留。两者切换只需 `LLM_MODEL` 环境变量，见下方"换模型"。

> **注意**：`ollama create` 需要服务有写模型目录的权限。如果报 `operation not permitted`，
> 在**自己的终端**里重启一次服务再执行：
> ```bash
> brew services restart ollama     # 或 pkill ollama && ollama serve
> ```

## 换模型

`app/config.py` 里的 `LLM_MODEL` 支持环境变量覆盖，换模型不用改代码：

```bash
# 用面壁更大的多模态版本（要能 pull 得动）
ollama pull minicpm-v
LLM_MODEL=minicpm-v streamlit run app/ui.py

# 或者任何本地已下载的模型
LLM_MODEL=qwen2.5:7b python -m app.benchmark
```

## 为什么选 2B 而不是 7B

这是有意的，不是将就：

- **端侧叙事要成立**：2B Q4 只占约 1.4GB 内存，一台 16GB 的普通办公本能同时开浏览器和办公软件跑它。换成 7B 就要 8-9GB，客户那边的办公电脑未必扛得住
- **政企场景的任务很窄**：制度问答是"检索到位 + 照抄条款"，不需要模型自己推理。检索质量比模型参数量更决定最终效果——这也是本项目把力气全花在切分、混合检索、拒答闸门上的原因
- **实测口径更诚实**：README 里写"2B 模型在 M1 上跑到 X 秒"，比写"7B 模型效果更好"更能说明端侧方案的可行性边界
