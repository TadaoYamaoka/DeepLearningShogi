from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lc0_upstream import modified_sources, upstream_source
from datetime import datetime, timezone
import hashlib, json
root=Path.cwd(); folder=root/'artifacts/fpu_review'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def logtext(p):
 data=Path(p).read_bytes()
 return data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
def read(name): return json.loads((folder/(name+'.json')).read_text(encoding='utf-8'))
variants=('ny','no_ny','onnx_ny','onnx_no_ny','makebook_ny','makebook_no_ny','multiponder_ny','multiponder_no_ny','tensorrt_double','tensorrt_debug')
builds={}
for v in variants:
 exe='dlshogi_onnxruntime' if v.startswith('onnx') else 'usi_multiponder' if v.startswith('multiponder') else 'dlshogi_tensorrt_double' if v=='tensorrt_double' else 'dlshogi_tensorrt'
 assert (folder/f'build_{v}.log').exists()
 builds[v]={'exe_sha256':sha(folder/v/(exe+'.exe')),'pdb_sha256':sha(folder/v/(exe+'.pdb'))}
for c in ('Release','Debug'):
 assert 'classic adapter tests passed' in logtext(folder/f'unit_{c}.log')
options={}
for v in (*variants[:6],'tensorrt_debug'):
 d=read('options_'+v)
 assert d['complete'] and len(d['checks'])==8
 assert d['engine_sha256']==builds[v]['exe_sha256']
 options[v]=d
nn={}
for v in variants[:6]:
 d=read('nn_'+v)
 assert d['complete'] and not d['requires_investigation'] and len(d['records'])==100
 assert d['new_sha256']==builds[v]['exe_sha256']
 nn[v]={k:d[k] for k in ('new_sha256','old_sha256','model_sha256')}
 nn[v]['positions']=len(d['records'])
protocol={v:read('protocol_'+v) for v in ('ny','onnx_ny')}
for v,d in protocol.items(): assert len(d['passed'])==21 and d['engine_sha256']==builds[v]['exe_sha256']
books={m:read('book_'+m) for m in ('ny','no_ny')}
for m,d in books.items():
 for key in ('readback','policy_blend','makebook_policy_blend','unchanged_book_cache_preserved','policy_toggle_invalidates_cache','same_path_book_edit_invalidates_cache'):assert d[key]
 assert d['engine_sha256']==builds['makebook_'+m]['exe_sha256'] and d['reader_sha256']==builds[m]['exe_sha256']
fpu={}
for v in (*variants[:6],'tensorrt_debug'):
 d=read('fpu_'+v)
 assert d['complete'] and len(d['checks'])==5 and len(d['searches'])==4
 assert d['engine_sha256']==builds[v]['exe_sha256']
 fpu[v]=d
equivalence={m:read('equivalence_'+m) for m in ('ny','no_ny')}
for m,d in equivalence.items():
 assert d['complete'] and len(d['records'])==30
 assert d['new_sha256']==builds[m]['exe_sha256']
 assert d['bestmove_difference_samples']==0 and d['pv_difference_samples']==0
 assert d['transposed_cache_policy_equal']
assert 'optimizer float FPU defaults and integer PUCT/temperature parameters passed' in logtext(folder/'optimizer.log')
multi=read('multiponder')
assert multi['hits']==10 and multi['misses']==10 and multi['children_exited']
assert multi['engine_sha256']==builds['multiponder_ny']['exe_sha256']
assert multi['auxiliary_sha256']==builds['ny']['exe_sha256']
for name in ('smoke_double','smoke_multiponder'):
 assert 'bestmove ' in logtext(folder/(name+'.log'))
copies=(('ny','usi/x64/Release','dlshogi_tensorrt'),('tensorrt_debug','usi/x64/Debug','dlshogi_tensorrt'),('tensorrt_double','usi/x64/Release_WinTypeDouble','dlshogi_tensorrt_double'),('makebook_ny','usi/x64/MakeBook','dlshogi_tensorrt'),('onnx_no_ny','usi_onnxruntime/x64/Release','dlshogi_onnxruntime'),('multiponder_no_ny','usi_multiponder/x64/Release','usi_multiponder'))
for v,d,n in copies:
 for ext in ('exe','pdb'): assert sha(root/d/(n+'.'+ext))==sha(folder/v/(n+'.'+ext))
changes=modified_sources(root)
assert len(changes)==6
sources=('cppshogi/usi.hpp','test/check_fpu_options.py','test/check_fpu_optimizer.py','test/classic_tests.cpp','dlshogi/utils/usi_params_optimizer.py','usi/main.cpp','lc0_shogi/backend.cpp','lc0_shogi/backend.h','lc0_shogi/search_session.cpp','lc0_shogi/search_session.h','cppshogi/usi.cpp','dlshogi/utils/benchmark.py','test/check_search_options.py','test/check_makebook.py','test/backend_concurrency.cpp')
report=dict(complete=True,verified_at=datetime.now(timezone.utc).isoformat(),physical_gpu_count=1,builds=builds,unit_configurations=['Release','Debug'],options=options,nn=nn,protocol=protocol,books=books,fpu=fpu,equivalence=equivalence,optimizer_mocked_test=True,strength_tuning_performed=False,multiponder=multi,smoke=['double','multiponder_no_ny'],default_executables_and_symbols_updated=True,lc0_upstream_modified_files=changes,performance_remeasured=False,source_sha256={p:sha(root/p) for p in sources})
(root/'test/fpu_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('FPU review passed: 10 builds, 600 NN positions, 35 FPU checks, 56 readiness checks, 42 USI checks, 60 fixed-position comparisons; defaults verified, no strength tuning')
