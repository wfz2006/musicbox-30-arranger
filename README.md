# 三十音八音盒编曲工具

将支持的钢琴 MusicXML 分析、改编并导出为三十音八音盒 MusicXML 和 MIDI。工具提供简洁与流动两种伴奏，并检查音高、速度、同音间隔及导出一致性，适合由能够运行命令行的 agent 调用。

当前是早期 MVP，旋律采用每个起音时刻取选定声部最高音的启发式，听感和其他曲目的泛化效果仍需实际评估。反复跳转、未定时装饰音等无法解释的结构会报告为不支持。

## 安装与运行

需要 Python 3.11+，并安装依赖：

```bash
python -m pip install -r requirements.txt
python musicbox.py analyze --input input.musicxml --out analysis-output
python musicbox.py arrange --input input.musicxml --out arrangement-output --texture both
python -B tests/test_mvp.py -v
```

输出目录应为新的空目录。编曲输出包括两版 `.musicxml`、`.mid`、事件与修改记录、校验报告。自动移调与时间比例通过省略相应参数启用。其他调用选项见 [agent 使用指南](agent/USAGE.md)。

## 乐器规则

实际音高仅允许：F3、G3、C4、D4、E4、F4、G4、A4、Bb4、B4、C5、Db5、D5、Eb5、E5、F5、Gb5、G5、Ab5、A5、Bb5、B5、C6、Db6、D6、Eb6、E6、F6、G6、A6。四分音符速度为 100–220 BPM，相同实际音高的相邻起音间距至少为一个四分音符。

本仓库不包含原始参考乐谱、用户作品或生成的成品乐谱。程序化检查不能代替试听、谱面审核及实体八音盒测试。
