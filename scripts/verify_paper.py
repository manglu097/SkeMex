"""Validate paper settings and optional processed split counts/checksums offline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medmem_agent.config import GlobalConfig
from medmem_agent.evolution.config import load_evo_config

def verify(train_root=None,test_root=None,strict_hash=False):
    errors=[]
    config=load_evo_config(ROOT/'configs/evolution/default.json',ROOT/'configs/evolution/offline.json')
    checks={
        'retrieval.top_k':6,'retrieval.lambda_similarity':.4,'retrieval.lambda_utility':.4,'retrieval.lambda_memory':.2,
        'retrieval.min_similarity_threshold':.2,'retrieval.prefer_mature_bonus_ratio':.1,
        'retrieval.embedding_model':'text-embedding-3-large','buffer.window_size':30,'buffer.capacity':20,
        'encode.model':'deepseek-v3.2','categories.classifier_model':'deepseek-v3.2',
        'retrieval.pre_screen.model':'deepseek-v3.2','encode.draft_initial_utility':.5,
        'utility.category_ema_alpha':.2,'utility.category_ema_min_samples':3,'utility.category_ema_default':.5,
        'utility.negative_base_penalty':.1,'utility.negative_harm_scale':.5,'utility.positive_advantage_scale':1.,
        'utility.utility_clip_min':0.,'utility.utility_clip_max':1.,
        'utility.lr_base':.05,'utility.lr_max':.2,'utility.lr_warmup_steps':5,'utility.lr_decay_steps':20,
        'governance.manage_every_n_windows':2,'governance.merge_similarity_threshold':.8,
        'governance.mature_utility_threshold':.75,'governance.mature_usage_threshold':15,
        'governance.capacity_general':12,'governance.capacity_task_level_per_category':8,'governance.capacity_action_level_per_tool':5,
        'context_guard.token_budget':16384,'context_guard.key_finding_pin_step':5,'context_guard.context_trim_ratio':.8,
        'context_guard.context_trim_keep_last_n_steps':3,'context_guard.context_trim_observation_chars':200,
        'offline.freeze_skill_library_during_test':True,
    }
    for key,expected in checks.items():
        actual=config
        for name in key.split('.'):actual=getattr(actual,name)
        if actual!=expected:errors.append(f'{key}: expected {expected!r}, got {actual!r}')
    if GlobalConfig.from_file(ROOT/'configs/global_config.json').max_steps!=7:errors.append('max_steps must be 7')
    online_config=load_evo_config(ROOT/'configs/evolution/default.json',ROOT/'configs/evolution/online.json')
    if online_config.online.epochs!=3:errors.append('Online epochs must be 3 (camera-ready Table 3)')
    manifest=json.loads((ROOT/'data/splits/manifest.json').read_text())
    id_benches={r['dataset'] for r in manifest['datasets'] if r['train']}
    if set(config.offline.selected_benches)!=id_benches:errors.append('Offline training selection must contain exactly the seven ID configurations')
    for split,root in [('train',train_root),('test',test_root)]:
        if root is None:continue
        total=0
        for item in manifest['datasets']:
            path=Path(root)/item['relative_file'];expected=item[split]
            if not path.exists():
                if expected:errors.append(f'{split}/{item["dataset"]}: missing file')
                continue
            raw=path.read_bytes();lines=[line for line in raw.splitlines() if line.strip()]
            total+=len(lines)
            if len(lines)!=expected:errors.append(f'{split}/{item["dataset"]}: expected {expected} records, found {len(lines)}')
            for n,line in enumerate(lines,1):
                try:json.loads(line)
                except (ValueError,UnicodeDecodeError):errors.append(f'{split}/{item["dataset"]}:{n}: invalid JSON')
            if strict_hash and expected and hashlib.sha256(raw).hexdigest()!=item['local_snapshot_sha256'].get(split):
                errors.append(f'{split}/{item["dataset"]}: checksum differs from inspected local snapshot')
        print(f'{split}: {total} records checked')
    for error in errors:print('FAIL:',error)
    print(f'Paper verification: {len(errors)} error(s); {len(checks)+3} configuration checks. Numerical results are not evaluated.')
    return errors

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root',type=Path)
    parser.add_argument('--test-root',type=Path)
    parser.add_argument('--strict-hash',action='store_true',help='Require byte identity with the inspected local data snapshot')
    args=parser.parse_args()
    raise SystemExit(bool(verify(args.train_root,args.test_root,args.strict_hash)))
