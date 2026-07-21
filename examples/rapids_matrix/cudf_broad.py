import sys
def step(m): print(f'[cudf_broad] {m}', flush=True)
passed=[]; failed=[]
def run(name, fn):
    step(name)
    try:
        fn(); passed.append(name); print(f'  OK   {name}', flush=True)
    except Exception as e:
        failed.append(name); print(f'  ERR  {name}: {type(e).__name__}: {str(e)[:140]}', flush=True)

import cudf, numpy as np
print('cudf', cudf.__version__, flush=True)
def _assert(c):
    if not c: raise AssertionError()
def _gb():
    g=df.groupby('k').agg({'x':'sum','y':'mean'}).sort_index()
    _assert(int(g.loc['a','x'])==10 and int(g.loc['b','x'])==7 and int(g.loc['c','x'])==4)
def _arith():
    d=df.copy(); d['z']=d['x']*2+1; _assert(int(d['z'].sum())==48)
def _merge():
    r=cudf.DataFrame({'k':['a','b','c'],'val':[100,200,300]})
    m=df.merge(r,on='k',how='left'); _assert(len(m)==6 and int(m[m['k']=='c']['val'].iloc[0])==300)
def _fill():
    d=cudf.DataFrame({'a':[1.0,None,3.0]}); d['a']=d['a'].fillna(0).astype('int64'); _assert(int(d['a'].sum())==4)
def _ijoin():
    a=cudf.DataFrame({'k':[1,2,3],'v':[1,2,3]}); b=cudf.DataFrame({'k':[2,3,4],'w':[20,30,40]})
    _assert(len(a.merge(b,on='k'))==2)
df = cudf.DataFrame({'k':['a','b','a','c','b','a'],'x':[1,2,3,4,5,6],'y':[10.,20.,30.,40.,50.,60.]})

run('reductions',   lambda: (_ for _ in ()).throw(AssertionError) if not (int(df['x'].sum())==21 and abs(float(df['y'].mean())-35)<1e-6 and int(df['x'].max())==6 and int(df['x'].min())==1) else None)
run('groupby+agg',  lambda: _gb())
run('sort_values',  lambda: _assert(int(df.sort_values('x',ascending=False)['x'].iloc[0])==6))
run('filter',       lambda: _assert(len(df[df['x']>3])==3))
run('arithmetic',   lambda: _arith())
run('string_upper', lambda: _assert(df['k'].str.upper().iloc[0]=='A'))
run('string_contains', lambda: _assert(int(df['k'].str.contains('a').sum())==3))
run('merge_left',   lambda: _merge())
run('cumsum',       lambda: _assert(int(df['x'].cumsum().iloc[-1])==21))
run('nunique',      lambda: _assert(int(df['k'].nunique())==3))
run('to_pandas',    lambda: _assert(int(df.to_pandas()['x'].sum())==21))
run('fillna_astype',lambda: _fill())
run('inner_join',   lambda: _ijoin())

print(f'[cudf_broad] SUMMARY passed={len(passed)} failed={len(failed)}', flush=True)
if failed: print('  FAILED: '+', '.join(failed), flush=True)
print('[cudf_broad] DONE', flush=True)
