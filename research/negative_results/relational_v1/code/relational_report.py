"""Summarize downloaded evidence on matched historical GLEEP accuracy sets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from .stats import pearsonr, kendall_tau_b


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('evidence',type=Path)
    ap.add_argument('--exact',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    def load(p): return json.loads(p.read_text(encoding='utf-8'))
    base=args.evidence
    report=load(base/'results/relational-20260906/full.json')
    rows=report['records']
    if args.exact:
        exact=load(args.exact)
        look={(x['model'],x['source'],x['classes']):x for x in exact['records']}
        for row in rows:
            extra=look[(row['model'],row['source'],row['classes'])]
            for metric in ['rpt_cosine_exact','relation_cosine_raw']:
                row['scores'][metric]=extra['scores'][metric]
    metrics=['rpt','relation_raw','negative_entropy','mutual_information','projected_cka32','minus_class_count']
    if args.exact: metrics+=['rpt_cosine_exact','relation_cosine_raw']
    accuracy={}
    for row in rows:
        model,source,k=row['model'],row['source'],row['classes']
        run='full-score-'+('r18' if model=='ResNet18' else 'r34')+'-'+('c10' if source=='CIFAR10' else 'imagenet')
        for metric in ['GLEEP','LEEP']:
            path=base/'results/recomputed'/run/'published'/model/source/(metric+'.json')
            row['scores'][metric]=load(path)[f'{k:03d}']
        for strategy in ['Finetune','Retrain']:
            key=(strategy,model,source)
            if key not in accuracy:
                path=base/'historical/EXP2/result'/strategy/model/source/'test_ACC/GLEEP_ACC.json'
                accuracy[key]={int(k):v for k,v in load(path).items()}
    metrics=['GLEEP','LEEP']+metrics
    correlations=[]
    for strategy in ['Finetune','Retrain']:
        for model,source in sorted({(r['model'],r['source']) for r in rows}):
            a=accuracy[(strategy,model,source)]
            subset=[r for r in rows if r['model']==model and r['source']==source and r['classes'] in a]
            for metric in metrics:
                x=[r['scores'][metric] for r in subset]; y=[a[r['classes']] for r in subset]
                correlations.append(dict(strategy=strategy,model=model,source=source,metric=metric,
                    n=len(subset),pearson=pearsonr(x,y),kendall=kendall_tau_b(x,y)))
    fixed=[]
    for strategy in ['Finetune','Retrain']:
        for k in range(2,101):
            subset=[r for r in rows if r['classes']==k and not r['reference_self']]
            if len(subset)!=3 or any(k not in accuracy[(strategy,r['model'],r['source'])] for r in subset): continue
            y=[accuracy[(strategy,r['model'],r['source'])][k] for r in subset]
            for metric in metrics:
                if metric=='minus_class_count': continue
                try: tau=kendall_tau_b([r['scores'][metric] for r in subset],y)
                except ValueError: continue
                fixed.append(dict(strategy=strategy,classes=k,metric=metric,kendall=tau))
    lines=['# 无聚类关系评分：EXP2 首轮实验结果','',
        '日期：2026-09-06。复用全部可用历史准确率，没有微调或训练下游模型。', '',
        '**结论：当前 RPT 未达到 GLEEP 的效果，不能作为等效替代方法。类别数趋势是本实验的重要混杂因素。**','',
        '## 实验口径','',
        '- 4组缓存 × 99个任务，共396个评分任务；RBF采样版每任务50000次采样，随机种子20260906。',
        '- 固定参考为 ResNet34/ImageNet，参考自身的99个评分标为诊断结果；主要评价应排除这组。',
        '- 所有方法在每个分支使用同一份 GLEEP_ACC.json 及其有效任务交集；64条缺失设置保持64对。',
        '- LEEP/GLEEP 对照使用服务器固定种子重算的 legacy 分数。此处 LEEP 的准确率配对统一到了 GLEEP 分支，可能与旧表的 LEEP 均值略不同。',
        '- 保留 published 分类头和数据语义。参考缓存样本顺序沿用统一提取流程，并核对标签序列及任务类别清单；旧缓存无逐样本ID，不能额外证明类内排列。',
        '- CKA 为固定32维随机投影后的线性 CKA，不能视为完整 CKA。类别数基线是混杂诊断，不能称为类别数未知的无监督方法。','',
        '## 跨任务相关性均值','',
        '| 方法 | 全8分支 Pearson | 全8分支 Kendall | 排除参考自身 Pearson | 排除参考自身 Kendall |',
        '| --- | ---: | ---: | ---: | ---: |']
    summary={}
    for metric in metrics:
        allrows=[r for r in correlations if r['metric']==metric]
        primary=[r for r in allrows if not(r['model']=='ResNet34' and r['source']=='ImageNet')]
        vals=[mean(r[t] for r in group) for group in [allrows,primary] for t in ['pearson','kendall']]
        summary[metric]=vals
        lines.append('| '+metric+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','## 固定类别数的跨模型诊断','',
        '利用现有准确率，在同一类别数下比较3个非参考模型，无需新增训练。每个任务只有3个候选，因此 Kendall 很粗；嵌套任务也不独立，这只是方向诊断，不是充分的模型库验证。','',
        '| 方法 | Finetune 平均 Kendall | 任务数 | Retrain 平均 Kendall | 任务数 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for metric in metrics:
        if metric=='minus_class_count':continue
        groups=[[r for r in fixed if r['metric']==metric and r['strategy']==s] for s in ['Finetune','Retrain']]
        lines.append(f'| {metric} | {mean(r["kendall"] for r in groups[0]):.6f} | {len(groups[0])} | {mean(r["kendall"] for r in groups[1]):.6f} | {len(groups[1])} |')
    lines+=['','## 时间与验证','',
        '- Slurm 598094：24任务试跑，8秒；598096：396任务全量运行，81秒。',
        f'- RBF RPT内部评分计时合计{sum(r["rpt_seconds"] for r in rows):.2f}秒，不含 softmax、其他基线、缓存读取和参考编码器前向。',
        '- 与历史19秒LEEP缓存批次相比，没有证据证明RBF版比LEEP更快。与GMM的速度优势还需统一运行环境后严格计时。',
        '- 4项关系评分数值测试通过，含精确矩阵对照、随机估计误差、纯自环零分和非法输入；既有2项LEEP测试也通过。',
        '- 未进行目标标签调参。余弦精确版是在第一轮结果之后追加的探索性核函数对照；它同时改变核函数，因此无法单独隔离RBF采样误差。','',
        '## 后续判断','',
        '不建议据此直接撰写新方法论文或声称达到GLEEP。先利用固定类别数的现有准确率识别真正的模型选择信号，再决定是否继续关系评分路线。需要单独检验源分类头限制及独立参考依赖；不得为追上跨任务Pearson直接乘以类别数惩罚并将其作为创新。','']
    if args.exact:
        lines+=['精确余弦版内部评分计时合计：'+f'{sum(r["rpt_seconds"] for r in exact["records"]):.2f}秒。该实现通过充分统计量计算，不构建全样本矩阵；复杂度为 O(n C_s d_r)。','']
    args.output.write_text('\n'.join(lines),encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(dict(summary=summary,correlations=correlations,fixed_class=fixed),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
