# 壁纸助手

自产自销的壁纸生成工具，可以训练模型。

> 需要安装 Python 3.8+，并确保已安装 Node.js 20+

## 安装

```bash
# 安装前端依赖
npm/yarn/pnpm install

# 安装 python 依赖
cd ai-module
pip install -r requirements.txt
```

## 生成

```bash
cd ai-module
python scripts/generate.py -f test_config.json
```

## 运行Electron应用

```bash
npm run dev
```