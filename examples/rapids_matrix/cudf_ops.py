import cudf
oks=[]; errs=[]
def run(name, fn):
    try: fn(); oks.append(name); print(f'OK  {name}', flush=True)
    except Exception as e: errs.append(name); print(f'ERR {name}: {type(e).__name__}: {str(e)[:90]}', flush=True)

df = cudf.DataFrame({'k':[1,2,1,3,2,1],'x':[1,2,3,4,5,6],'y':[10.,20.,30.,40.,50.,60.]})
run('reductions', lambda: (_ for _ in ()).throw(AssertionError()) if not (int(df['x'].sum())==21 and abs(float(df['y'].mean())-35)<1e-6 and int(df['x'].max())==6 and int(df['x'].min())==1) else None)
run('filter', lambda: (_ for _ in ()).throw(AssertionError()) if len(df[df['x']>3])!=3 else None)
def _arith():
    d=df.copy(); d['z']=d['x']*2+1; assert int(d['z'].sum())==48
run('arithmetic', _arith)
def _gb_sum():
    g=df.groupby('k').agg({'x':'sum'}).to_pandas().sort_index(); assert int(g.loc[1,'x'])==10 and int(g.loc[2,'x'])==7 and int(g.loc[3,'x'])==4
run('groupby_sum', _gb_sum)
def _gb_multi():
    g=df.groupby('k').agg({'x':'sum','y':'mean'}).to_pandas().sort_index(); assert int(g.loc[1,'x'])==10
run('groupby_sum+mean', _gb_multi)
def _gb_count():
    g=df.groupby('k').agg({'x':'count'}).to_pandas().sort_index(); assert int(g.loc[1,'x'])==3
run('groupby_count', _gb_count)
def _merge():
    r=cudf.DataFrame({'k':[1,2,3],'v':[100,200,300]}); m=df.merge(r,on='k',how='left'); assert len(m)==6
run('merge_join', _merge)
def _sortvals():
    s=df.sort_values('x',ascending=False); assert int(s['x'].iloc[0])==6
run('sort_values', _sortvals)
def _cumsum():
    assert int(df['x'].cumsum().iloc[-1])==21
run('cumsum', _cumsum)
def _nunique():
    assert int(df['k'].nunique())==3
run('nunique', _nunique)
def _topandas():
    assert int(df.to_pandas()['x'].sum())==21
run('to_pandas', _topandas)
print(f'SUMMARY ok={len(oks)} err={len(errs)}', flush=True)
if errs: print('FAILED: '+', '.join(errs), flush=True)
print('DONE', flush=True)
