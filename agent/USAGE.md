# Agent 使用指南

安装 `requirements.txt` 后，从仓库根目录运行：

```bash
python musicbox.py analyze --input path/to/piano.musicxml --out new-analysis-dir
python musicbox.py arrange --input path/to/piano.musicxml --out new-output-dir --texture both
```

`--texture` 可选 `sparse`、`flowing`、`both`。可指定 `--melody-part P1`、`--transpose 6` 或 `--time-scale 2`；省略移调或时间比例时自动搜索。输出目录必须是不存在的目录或空目录。

命令返回 JSON：退出码 0 表示成功，1 表示无可行方案或导出未通过检查，2 表示输入、依赖或支持范围问题。查看 `analysis.json`、`candidates.json`、`changes.json` 和 `validation.json` 了解旋律选择、改动及限制。MusicXML 和 MIDI 从同一事件表生成，机械检查通过不代表听感已验收。
