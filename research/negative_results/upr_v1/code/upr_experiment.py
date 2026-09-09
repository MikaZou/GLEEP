"""EXP2 benchmark adapter. Labels/accuracies are confined to this evaluation layer."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import pickle
import platform
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from .config import default_exp2_cache_dir, default_result_dir, runtime_data_root, exp2_results_root, workspace_root
from .exp2 import selected_classes, parse_tasks, _filtered_dataset, _source_outputs
from .io_utils import load_json, write_json, write_text, sha256, markdown_table
from .metrics import softmax, leep_from_logits, gleep_from_logits
from .stats import pearsonr, kendall_tau_b
from .upr import project_features, score_upr, structural_diagnostics, linear_cka, rankme

KEYS = ['ResNet18/CIFAR10', 'ResNet18/ImageNet', 'ResNet34/CIFAR10', 'ResNet34/ImageNet']
OLD_RUNS = ['full-score-r18-c10', 'full-score-r18-imagenet', 'full-score-r34-c10', 'full-score-r34-imagenet']


def correlations(x, y):
    out = {"n": len(x)}
    for name, function in [("pearson", pearsonr), ("kendall", kendall_tau_b)]:
        try:
            out[name] = function(x, y)
        except ValueError:
            out[name] = None
    return out


def load_pool(cache, data_root):
    # Verify official pickle checksum BEFORE loading the trusted dataset file.
    test_file = data_root / 'cifar-100-python' / 'test'
    if hashlib.md5(test_file.read_bytes()).hexdigest() != 'f0ef6b0ae62326f3e7ffdfab6717acfc':
        raise ValueError('CIFAR100 test checksum mismatch')
    with test_file.open('rb') as stream:
        raw = pickle.load(stream, encoding='bytes')
    labels = np.asarray(raw[b'fine_labels'])
    ids = [hashlib.sha256(row.tobytes()).hexdigest() for row in raw[b'data']]
    audit = {"raw_test_sha256": sha256(test_file), "sample_count": len(labels),
             "sample_order_digest": hashlib.sha256(''.join(ids).encode()).hexdigest(),
             "identity_evidence": "reconstructed from official test row order + deterministic extraction source; original caches have no per-image IDs",
             "identity_limitation": "not an independent cryptographic binding of each activation row to an image",
             "filter_source": inspect.getsource(_filtered_dataset),
             "forward_source": inspect.getsource(_source_outputs), "groups": []}
    pool = {}
    for key in KEYS:
        root = cache / 'published' / key
        manifest = load_json(root / 'manifest.json')
        arrays = {}
        for name, a in manifest['arrays'].items():
            path = root / a['path']
            if sha256(path) != a['sha256']:
                raise ValueError(f'cache checksum mismatch: {path}')
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(arrays[name].shape) != a['shape'] or not np.isfinite(arrays[name]).all():
                raise ValueError(f'invalid cache array: {path}')
        if not np.array_equal(labels, arrays['labels']) or len(arrays['features']) != len(ids):
            raise ValueError(f'row-order/label mismatch: {key}')
        if manifest['split'] != 'test' or manifest['semantics'] != 'published':
            raise ValueError('unexpected cache semantics')
        pool[key] = arrays
        audit['groups'].append({"key": key, "manifest_sha256": sha256(root/'manifest.json'),
                                "labels_match_raw_order": True, "checkpoint": manifest['source_checkpoint_sha256']})
    return pool, labels, audit


def counterexamples():
    rng = np.random.default_rng(314)
    n = 240
    labels = np.arange(n) % 2
    nuisance = rng.normal(size=(n, 4))
    cases = {}
    for case in ['shared_semantic', 'shared_background', 'independent_noise', 'constant']:
        pool = {}
        for key in KEYS:
            if case == 'shared_semantic':
                x = (2*labels[:,None]-1)*rng.normal(size=(1,16)) + .08*rng.normal(size=(n,16))
            elif case == 'shared_background':
                x = nuisance @ rng.normal(size=(4,16)) + .02*rng.normal(size=(n,16))
            elif case == 'constant':
                x = np.ones((n,16))
            else:
                x = rng.normal(size=(n,16))
            pool[key] = project_features(x)
        result, ridge, v, _ = score_upr(pool, KEYS[0])
        diag = structural_diagnostics(ridge, v, labels)
        permuted = structural_diagnostics(ridge, v, rng.permutation(labels))
        cases[case] = {**result, **diag, "permuted_labels_loo_accuracy": permuted['ridge_loo_accuracy'],
                       "score_unchanged_by_label_permutation": True}
    return cases


def summarize(records, accuracy_root):
    rows, within, oracle, historical = [], [], [], []
    seeds = sorted({r['seed'] for r in records})
    for seed in seeds:
        chosen = [r for r in records if r['seed'] == seed]
        methods = list(chosen[0]['scores'])
        for strategy in ['Finetune', 'Retrain']:
            accuracies = {key: load_json(accuracy_root/strategy/key/'test_ACC/LEEP_ACC.json') for key in KEYS}
            for key in KEYS:
                rr = sorted([r for r in chosen if r['key'] == key], key=lambda r: r['task'])
                for metric in methods:
                    x = [r['scores'][metric] for r in rr]
                    y = [float(accuracies[key][r['task']]) for r in rr]
                    rows.append({"seed": seed, "strategy": strategy, "key": key, "metric": metric,
                                 "accuracy_source": str(accuracy_root/strategy/key/'test_ACC/LEEP_ACC.json'),
                                 **correlations(x,y)})
                own = load_json(accuracy_root/strategy/key/'test_ACC/GLEEP_ACC.json')
                paired = [r for r in rr if r['task'] in own]
                historical.append({"seed":seed,"strategy":strategy,"key":key,
                                   "metric":"gleep_own_accuracy", **correlations(
                                       [r['scores']['gleep'] for r in paired], [float(own[r['task']]) for r in paired])})
            for task in sorted({r['task'] for r in chosen}):
                rr = sorted([r for r in chosen if r['task']==task], key=lambda r:r['key'])
                for metric in methods:
                    within.append({"seed":seed,"strategy":strategy,"task":task,"metric":metric,
                                   **correlations([r['scores'][metric] for r in rr],
                                                  [float(accuracies[r['key']][task]) for r in rr])})
        if seed == 0:
            for key in KEYS:
                rr = [r for r in chosen if r['key']==key]
                oracle.append({"key":key,"upr_vs_negative_ridge_risk":correlations(
                    [r['scores']['upr'] for r in rr],[-r['diagnostics']['ridge_loo_risk'] for r in rr]),
                    "upr_vs_ridge_accuracy":correlations([r['scores']['upr'] for r in rr],
                        [r['diagnostics']['ridge_loo_accuracy'] for r in rr])})
    means=[]
    for seed in seeds:
        for metric in list(records[0]['scores']):
            r=[x for x in rows if x['seed']==seed and x['metric']==metric]
            w=[x for x in within if x['seed']==seed and x['metric']==metric]
            item={"seed":seed,"metric":metric}
            for name,group in [('across_tasks',r),('within_k',w)]:
                for coeff in ['pearson','kendall']:
                    valid=[x[coeff] for x in group if x[coeff] is not None]
                    item[name+'_'+coeff]=float(np.mean(valid)) if valid else None
                    item[name+'_'+coeff+'_valid']=len(valid)
            means.append(item)
    return {"correlations":rows,"within_k":within,"means":means,"oracle":oracle,"historical_accuracy":historical}


def render(report, destination):
    summary=report['summary']
    def fmt(x): return 'undefined' if x is None else f'{x:.6f}'
    lines=['# UPR EXP2 概念验证', '', '参数在运行前固定；真实标签仅用于任务适配、监督基线和理论诊断。', '',
           '## 统一准确率的结果', '',
           '所有方法使用同一份 LEEP_ACC.json。GLEEP 分数来自已有固定种子全量重算；缓存 GMM 复算另用于计时和数值对照。', '',
           markdown_table(['Seed','Method','平均 Pearson','平均 Kendall','固定 k Pearson','固定 k Kendall'],
               [(r['seed'],r['metric'],fmt(r['across_tasks_pearson']),fmt(r['across_tasks_kendall']),
                 fmt(r['within_k_pearson']),fmt(r['within_k_kendall'])) for r in summary['means']]), '',
           '跨任务系数先在每组99个任务内计算，再对4组模型/源数据×2种训练方式等权平均；固定 k 则在4个模型间计算，再对99个k×2种训练方式等权平均。Retrain为历史名称，实际是冻结骨干的linear probing。', '',
           '固定 k 的 1/k 和 1/n 分数为常数，相关性未定义，不能填写成0。RankMe保留有效秩越高分数越高的原方向；不根据本次结果翻转符号。', '',
           '## 分组主结果与历史口径', '',
           markdown_table(['训练','模型/源数据','方法','任务数','Pearson','Kendall'],
               [(r['strategy'],r['key'],r['metric'],r['n'],fmt(r['pearson']),fmt(r['kendall']))
                for r in summary['correlations'] if r['seed']==0 and r['metric'] in ['upr','gleep']]), '',
           '以下另将确定性GLEEP分数关联到历史GLEEP_ACC.json；这不是共同accuracy主表，也不是原论文无种子分数的逐项复刻。', '',
           markdown_table(['训练','模型/源数据','实际任务数','Pearson','Kendall'],
               [(r['strategy'],r['key'],r['n'],fmt(r['pearson']),fmt(r['kendall']))
                for r in summary['historical_accuracy'] if r['seed']==0]), '',
           'omit_X表示从每个候选的参考池中删除X；当X就是被评模型时，该行不变。不同候选的参考目标本来就不同，消融导致的模型排序变化还可能来自参考难度变化，不能自动解释为更准确的语义估计。', '',
           '## 理论诊断', '']
    diag=[r['diagnostics'] for r in report['records'] if r['seed']==0]
    lines += [f"检查 {len(diag)} 个真实任务：非零条件准确率下界 {sum(d['bound_nonvacuous'] for d in diag)} 个；风险界通过 {sum(d['risk_bound_holds'] for d in diag)} 个。",
              '', 'epsilon 使用真实标签事后计算，是 oracle 诊断；部署时不可获得。此界仅针对固定设计岭回归留一分类，不是Finetune泛化保证。', '',
              markdown_table(['模型/源数据','UPR vs -ridge风险 P','UPR vs -ridge风险 K','UPR vs ridge准确率 P','UPR vs ridge准确率 K'],
                  [(r['key'],fmt(r['upr_vs_negative_ridge_risk']['pearson']),fmt(r['upr_vs_negative_ridge_risk']['kendall']),
                    fmt(r['upr_vs_ridge_accuracy']['pearson']),fmt(r['upr_vs_ridge_accuracy']['kendall'])) for r in summary['oracle']]), '',
              '## 反例', '', markdown_table(['场景','UPR','岭留一准确率','打乱标签后准确率','下界'],
                  [(k,fmt(v['score']),fmt(v['ridge_loo_accuracy']),fmt(v['permuted_labels_loo_accuracy']),fmt(v['accuracy_lower_bound']))
                   for k,v in report['counterexamples'].items()]), '', '## 计时与门槛', '',
              f"预处理耗时（各seed）：{report['projection_seconds']} 秒。主实验总耗时 {report['experiment_seconds']:.3f} 秒，包含基线、消融和oracle诊断。",
              '', '一次性投影计入总成本；每次UPR计时包含参考构建及留一求解。计时使用同一节点、同一线程数及缓存输入。',
              '参考带宽依赖当前任务，参考构建是每任务成本；不声称它可跨99个任务一次构建。score_upr还计算当前模型的RFF供消融复用，但主分数只使用另外三个模型。', '',
              f"计时协议：{report.get('timing_protocol', {}).get('description', '原始运行：首个GMM计时含sklearn惰性导入，须以复核计时为准')}。", '']
    timings=report.get('timings',[])
    if timings:
        lines += [markdown_table(['组','k','UPR秒','GMM秒','GMM/UPR','GLEEP缓存差'],
                  [(r['key'],r['task'],f"{r['upr_seconds']:.6f}",f"{r['gmm_seconds']:.6f}",
                    f"{r['gmm_seconds']/r['upr_seconds']:.2f}",f"{r['gleep_delta']:.3g}") for r in timings]), '']
    main={r['metric']:r for r in summary['means'] if r['seed']==0}
    parity=all(main['upr']['across_tasks_'+c] >= main['gleep']['across_tasks_'+c]-.01 for c in ['pearson','kendall'])
    speed=float(np.median([r['gmm_seconds']/r['upr_seconds'] for r in timings])) if timings else None
    report['gates']={"correlation_parity":parity,"sampled_median_speedup":speed,
                     "sampled_speed_gate":bool(speed and speed>=10),
                     "bound_nonvacuous_count":sum(d['bound_nonvacuous'] for d in diag),
                     "timing_scope":"12 predefined endpoints, not full 396-task GMM timing"}
    lines += [f"相关性门槛：{'PASS' if parity else 'FAIL'}。抽样中位加速比：{fmt(speed)}。",
              '', '计时覆盖预先指定的 k=2,10,100，不能外推为396任务全量实测时间。', '',
              '## 决策', '',
              ('当前原型未同时满足相关性、有效理论界和基线优势，不支持作为成熟GLEEP替代方法。保留负结果，停止将本版本包装为论文改进。'
               if not parity or not any(d['bound_nonvacuous'] for d in diag) else
               '已通过部分门槛，仍需固定类别数排序、模型池稳健性与独立数据集验证后判断论文贡献。'), '',
              '## 适用边界', '',
              '- 共识可能保留共享背景；排除自身并不保证参考模型误差独立。',
              '- 随机特征/带宽使用全部无标签任务样本，留一恒等式固定这些预处理；它不是全流程严格归纳留一验证。',
              '- 原始缓存缺少逐图像ID，样本关联依据官方test顺序、确定性提取源码和数组哈希，不能声称独立逐图像认证。',
              '- 99个嵌套类别任务不独立；不把它们的普通bootstrap当作独立任务置信区间。',
              '- seed1/2是随机近似稳定性，不是新任务或新下游训练。', '']
    write_text(destination/'UPR_RESULTS.md','\n'.join(lines))
    write_json(destination/'report.json',report)


def benchmark_timings(report, pool, labels, destination, threads):
    # sklearn must already be imported before entering threadpool_limits in main.
    # Warm up initialization outside both timers; no image forward is performed.
    warm = np.random.default_rng(0).normal(size=(32,4))
    gleep_from_logits(warm, num_clusters=2, random_state=0)
    report['timing_protocol'] = {
        'description': 'sklearn预导入、合成GMM预热；UPR重复3次取中位数，GMM各1次；双方缓存已读入、投影成本另报',
        'host': platform.node(), 'threads': threads, 'threadpools': threadpool_info(),
        'benchmark_code_sha256': sha256(Path(__file__))}
    timings=[]
    projected={key:project_features(pool[key]['features'],0) for key in KEYS}
    for task in [2,10,100]:
        indices=np.flatnonzero(np.isin(labels,selected_classes(task,42,'published')))
        task_pool={key:x[indices] for key,x in projected.items()}
        for key in KEYS:
            times=[]
            for _ in range(3):
                t=time.perf_counter(); score_upr(task_pool,key); times.append(time.perf_counter()-t)
            print(f'GMM timing {key} k={task}',flush=True)
            cluster=pool[key]['clustering_logits'][indices]; prediction=pool[key]['prediction_logits'][indices]
            t=time.perf_counter()
            value,_=gleep_from_logits(cluster,prediction,num_clusters=task,random_state=0)
            elapsed=time.perf_counter()-t
            baseline=next(r['scores']['gleep'] for r in report['records']
                          if r['seed']==0 and r['key']==key and r['task']==f'{task:03d}')
            timings.append({'key':key,'task':task,'upr_seconds':float(np.median(times)),
                            'gmm_seconds':elapsed,'gleep_cached':value,
                            'gleep_delta':abs(value-baseline)})
            report['timings']=timings
            render(report,destination)
    report['complete']=True
    render(report,destination)
    write_json(destination/'progress.json',{'complete':True,'record_count':len(report['records'])})
    print(report['gates'],flush=True)


def retime(args):
    if args.output_dir.exists():
        raise ValueError('retiming requires a fresh output directory')
    report=load_json(args.retime_report)
    pool, labels, audit=load_pool(args.cache_dir,args.data_root)
    prior_audit=load_json(args.retime_report.parent/'alignment.json')
    if audit != prior_audit:
        raise ValueError('cache or extraction-source provenance changed')
    report['score_origin']={'report':str(args.retime_report),'sha256':sha256(args.retime_report),
                            'note':'scores unchanged; timing-only audit of lazy imports and thread limits'}
    report['complete']=False
    report.pop('timings',None)
    write_json(args.output_dir/'alignment.json',audit)
    write_json(args.output_dir/'protocol.json',report['protocol'])
    benchmark_timings(report,pool,labels,args.output_dir,args.threads)


def run(args):
    destination=args.output_dir
    if (destination/'report.json').exists():
        raise ValueError('result exists; select a fresh --output-dir')
    pool, labels, audit=load_pool(args.cache_dir,args.data_root)
    write_json(destination/'alignment.json',audit)
    fixed={"dimension":64,"rff_dimension":64,"ridge_strength":.01,"intercept":"penalized",
           "seeds":args.seeds,"tasks":args.tasks,"pool":KEYS,"threads":args.threads,
           "task_selection_seed":42,"sigma":"median positive sampled pair distance within task",
           "candidate_excluded":True,"host":platform.node(),"python":platform.python_version(),
           "threadpools":threadpool_info(),"code_sha256":sha256(Path(__file__))}
    write_json(destination/'protocol.json',fixed)
    old={key:load_json(args.old_root/name/'published'/key/'GLEEP.json') for key,name in zip(KEYS,OLD_RUNS)}
    records=[]
    projection_seconds={}
    start=time.perf_counter()
    for seed in args.seeds:
        t=time.perf_counter()
        projected={key:project_features(pool[key]['features'],seed) for key in KEYS}
        projection_seconds[str(seed)]=time.perf_counter()-t
        for task in parse_tasks(args.tasks):
            classes=selected_classes(task,42,'published')
            indices=np.flatnonzero(np.isin(labels,classes))
            task_pool={key:x[indices] for key,x in projected.items()}
            for key in KEYS:
                t=time.perf_counter()
                result,ridge,v,blocks=score_upr(task_pool,key,seed)
                elapsed=time.perf_counter()-t
                scores={"upr":result['score'],"gleep":float(old[key][f'{task:03d}'])}
                if seed==0:
                    pred=pool[key]['prediction_logits'][indices]
                    scores['leep']=leep_from_logits(pred,labels[indices])
                    scores['canonical_leep']=leep_from_logits(pred,labels[indices],formula='canonical')
                    p=softmax(pred)
                    scores['negative_entropy']=float(np.mean(np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)))
                    scores['rankme']=rankme(pool[key]['features'][indices])
                    scores['cka']=linear_cka(task_pool[key],v)
                    scores['in_sample']=-float(np.sum(ridge.residual(v,loo=False)**2)/len(indices))
                    all_v=np.concatenate([blocks[k] for k in sorted(KEYS)],axis=1)/2
                    scores['include_self']=-float(np.sum(ridge.residual(all_v)**2)/len(indices))
                    for omitted in KEYS:
                        refs=[k for k in result['references'] if k!=omitted]
                        vv=np.concatenate([blocks[k] for k in refs],axis=1)/np.sqrt(len(refs))
                        scores['omit_'+omitted]=-float(np.sum(ridge.residual(vv)**2)/len(indices))
                    scores['inverse_k']=1/task
                    scores['inverse_n']=1/len(indices)
                    diagnostics=structural_diagnostics(ridge,v,labels[indices])
                else:
                    diagnostics={}
                records.append({"key":key,"task":f'{task:03d}',"seed":seed,"n":len(indices),
                                "scores":scores,"upr_seconds":elapsed,"upr":result,"diagnostics":diagnostics})
            if task%10==0 or task==2:
                print(f'seed={seed} k={task} records={len(records)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
            write_json(destination/'progress.json',{"complete":False,"records":records})
    # Summaries per seed, since stability runs need only UPR/GLEEP.
    parts=[summarize([r for r in records if r['seed']==seed],exp2_results_root(workspace_root())) for seed in args.seeds]
    summary={key:sum([p[key] for p in parts],[]) for key in parts[0]}
    report={"protocol":fixed,"records":records,"projection_seconds":projection_seconds,
            "experiment_seconds":time.perf_counter()-start,"summary":summary,"counterexamples":counterexamples()}
    render(report,destination)
    benchmark_timings(report,pool,labels,destination,args.threads)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,default=default_exp2_cache_dir())
    p.add_argument('--data-root',type=Path,default=runtime_data_root())
    p.add_argument('--old-root',type=Path,default=default_result_dir())
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--tasks',default='2:100')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--retime-report',type=Path,help='reuse frozen scores, write audited timing to a NEW directory')
    args=p.parse_args()
    if 0 not in args.seeds:
        p.error('seed0 required for primary comparison')
    # Libraries loaded after a threadpool_limits context may evade its limits.
    from sklearn.mixture import GaussianMixture  # noqa: F401
    with threadpool_limits(limits=args.threads):
        retime(args) if args.retime_report else run(args)


if __name__=='__main__':
    main()
