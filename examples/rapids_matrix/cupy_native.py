import cudf, cupy
def run(name, fn):
    try: fn(); print(f'OK  {name}', flush=True)
    except Exception as e: print(f'ERR {name}: {type(e).__name__}: {str(e)[:90]}', flush=True)
print('cupy', cupy.__version__, flush=True)
def _cp():
    assert int(cupy.asarray([1,2,3]).sum()) == 6
run('cupy.asarray + sum', _cp)
df = cudf.DataFrame({'k':[3,1,2,1],'x':[1,2,3,4]})
def _vals():
    assert int(df['x'].values.sum()) == 10
run('cudf .values (cupy array)', _vals)
def _tocupy():
    assert int(df['x'].to_cupy().sum()) == 10
run('cudf .to_cupy()', _tocupy)
def _sortidx():
    g = df.groupby('k').agg({'x':'sum'}).sort_index()   # cudf-native sort_index (uses cupy)
    assert g.to_arrow().column('x').to_pylist() == [6,3,1]
run('cudf.sort_index (native, no pandas)', _sortidx)
print('DONE', flush=True)
