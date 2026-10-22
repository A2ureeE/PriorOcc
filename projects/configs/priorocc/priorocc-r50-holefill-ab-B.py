# 补洞 A/B —— B 臂（补洞 OFF）
# 与 A 臂唯一差异：enable_sem_continuity=False。其余（lr 1e-5、clip 5、
# warmup 200、全量帧、单帧路径、load_from=ep21）完全继承 A 臂，保证
# 差值只能归因到补洞 loss。
_base_ = ['./priorocc-r50-holefill-ab-A.py']

model = dict(
    enable_sem_continuity=False,
    sem_continuity=None,
)
