"""Post-run integrity and structure diagnostics; never tunes the UPR score."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import platform
import time

import numpy as np
from scipy.stats import pearsonr, kendalltau
from threadpoolctl import threadpool_limits, threadpool_info

from .config import default_exp2_cache_dir, runtime_data_root, exp2_results_root, workspace_root
from .exp2 import selected_classes
from .io_utils import load_json, write_json, write_text, sha256, markdown_table
from .upr import project_features, fourier_features, RidgeLOO, score_upr
from .upr_experiment import KEYS, load_pool


def review(report_path, output_dir):
    if output_dir.exists():
        raise ValueError('choose a fresh review directory')
    report = load_json(report_path)
    assert report.get('complete') is True, 'experiment not complete'
    records = report['records']
    identities = {(r['seed'],r['key'],r['task']) for r in records}
    expected = {(s,k,f'{t:03d}') for s in [0,1,2] for k in KEYS for t in range(2,101)}
    assert identities == expected and len(records) == len(expected)
    assert all(r['n'] == 100*int(r['task']) for r in records)
    assert all(np.isfinite(list(r['scores'].values())).all() for r in records)
    assert all(r['upr']['min_loo_denominator'] > 0 for r in records)
    assert all(r['diagnostics']['risk_bound_holds'] and r['diagnostics']['classification_bound_holds']
               for r in records if r['seed'] == 0)
    origin_path = report_path.parent
    score_origin_unchanged = None
    if 'score_origin' in report:
        from pathlib import Path
        origin_path = Path(report['score_origin']['report']).parent
        assert sha256(origin_path/'report.json') == report['score_origin']['sha256']
        origin = load_json(origin_path/'report.json')
        score_origin_unchanged = origin['records'] == records and origin['summary'] == report['summary']
        assert score_origin_unchanged

    accuracy_root = exp2_results_root(workspace_root())
    input_hashes = {}
    checked_correlations = 0
    for row in report['summary']['correlations']:
        path = accuracy_root/row['strategy']/row['key']/'test_ACC/LEEP_ACC.json'
        input_hashes[str(path)] = sha256(path)
        accuracy = load_json(path)
        assert len(accuracy) == 99
        rr = sorted([r for r in records if r['seed']==row['seed'] and r['key']==row['key']],key=lambda r:r['task'])
        x = [r['scores'][row['metric']] for r in rr]
        y = [accuracy[r['task']] for r in rr]
        np.testing.assert_allclose([pearsonr(x,y).statistic,kendalltau(x,y).statistic],
                                   [row['pearson'],row['kendall']],atol=1e-12)
        checked_correlations += 1

    pool, labels, audit = load_pool(default_exp2_cache_dir(),runtime_data_root())
    assert audit == load_json(report_path.parent/'alignment.json')
    t = time.perf_counter()
    projected = {key:project_features(pool[key]['features'],0) for key in KEYS}
    projection_seconds = time.perf_counter()-t
    rows = []
    for task in [2,10,100]:
        indices = np.flatnonzero(np.isin(labels,selected_classes(task,42,'published')))
        task_pool = {key:x[indices] for key,x in projected.items()}
        yy = labels[indices]
        _,encoded = np.unique(yy,return_inverse=True)
        y = np.eye(task)[encoded]
        n = len(indices)
        t = time.perf_counter()
        blocks = {key:fourier_features(task_pool[key],1009*(i+1))[0] for i,key in enumerate(sorted(KEYS))}
        construction_seconds = time.perf_counter()-t
        for key in KEYS:
            first,*_ = score_upr(task_pool,key)
            second,*_ = score_upr(task_pool,key)
            assert first == second, 'fixed-seed repeat differs'
            record = next(r for r in records if r['seed']==0 and r['key']==key and int(r['task'])==task)
            np.testing.assert_allclose(first['score'],record['scores']['upr'],atol=1e-12,rtol=0)
            t = time.perf_counter()
            v = np.concatenate([blocks[k] for k in sorted(KEYS) if k!=key],axis=1)/np.sqrt(3)
            ridge = RidgeLOO(task_pool[key])
            risk = float(np.sum(ridge.residual(v)**2)/n)
            solve_seconds = time.perf_counter()-t
            np.testing.assert_allclose(-risk,first['score'],atol=1e-12,rtol=0)
            total_kernel_sum = float(np.sum(v.sum(axis=0)**2))
            same_kernel_sum = float(np.sum((y.T@v)**2))
            same_pair_count = float(np.sum(y.sum(axis=0)**2))
            diag_sum = float(np.sum(v*v))
            rows.append({'key':key,'task':task,'n':n,'upr':first['score'],
                'fixed_seed_repeat_exact':True,'matches_frozen_score':True,
                'mean_kernel':total_kernel_sum/n**2,
                'mean_label_relation':same_pair_count/n**2,
                'same_class_offdiagonal_kernel':(same_kernel_sum-diag_sum)/(same_pair_count-n),
                'different_class_kernel':(total_kernel_sum-same_kernel_sum)/(n**2-same_pair_count),
                'epsilon_mean_direction_lower_bound':abs(total_kernel_sum-same_pair_count)/n,
                'epsilon_oracle':record['diagnostics']['epsilon_oracle'],
                'reference_pool_seconds_shared':construction_seconds,
                'ridge_score_seconds_after_reference':solve_seconds})

    result = {'report_sha256':sha256(report_path),'host':platform.node(),'python':platform.python_version(),
              'package_versions':{k:version(k) for k in ['numpy','scipy','scikit-learn','torch','torchvision','threadpoolctl']},
              'threadpools':threadpool_info(),'record_count':len(records),'independent_scipy_correlation_checks':checked_correlations,
              'score_origin_unchanged':score_origin_unchanged,'accuracy_sha256':input_hashes,
              'projection_seconds':projection_seconds,'structure_diagnostics':rows,
              'scope':'12 predefined endpoints; post-hoc diagnostics only, no score or parameter changes'}
    write_json(output_dir/'review.json',result)
    lines = ['# UPR 结果复核与结构诊断','',
        f"已检查{len(records)}条记录的完整性及有限性，使用SciPy独立复算{checked_correlations}组相关性；12个预设端点固定种子重复完全一致，并与原始分数在1e-12绝对误差内一致。",'',
        f"计时复核前后全部原始评分及相关性保持不变：{score_origin_unchanged}。",'',
        '## 参考结构与真实类别关系','',
        markdown_table(['模型/源','k','同类核均值(不含自身)','异类核均值','Q全体均值','A全体均值','epsilon'],
            [(r['key'],r['task'],f"{r['same_class_offdiagonal_kernel']:.6f}",f"{r['different_class_kernel']:.6f}",
              f"{r['mean_kernel']:.6f}",f"{r['mean_label_relation']:.6f}",f"{r['epsilon_oracle']:.3f}") for r in rows]),'',
        'A为真实one-hot同类关系。类间核值明显不为0时，Q不接近A；全体均值差乘以n还是谱误差的一个确定性下界。注意截距可部分消去共同分量，因此大谱误差也可能体现所用统一谱范数界过于保守，不能仅据此断言所有更精细的理论控制都不可能。', '',
        '## 分阶段成本（补充测量）','',
        f"一次性四模型投影：{projection_seconds:.6f}秒。下表参考池由四模型共建，可供同一任务四个候选复用；不能将该共享成本重复乘四，也不能把它与主计时中每个候选独立构建的成本混为一谈。",'',
        markdown_table(['模型/源','k','四模型参考构建(共享)秒','已知参考后岭评分秒'],
            [(r['key'],r['task'],f"{r['reference_pool_seconds_shared']:.6f}",f"{r['ridge_score_seconds_after_reference']:.6f}") for r in rows]),'',
        '这是独立诊断节点的一次分阶段测量，不用于重新计算正式GMM加速比；主报告仍使用同节点配对的整体评分时间。环境版本和输入accuracy哈希见review.json。','']
    write_text(output_dir/'UPR_REVIEW.md','\n'.join(lines))
    print({'checked_records':len(records),'checked_correlations':checked_correlations,'repeated_endpoints':len(rows)},flush=True)


def main():
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        review(args.report,args.output_dir)


if __name__=='__main__':
    main()
