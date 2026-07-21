import cudf
def run(name, fn):
    try: fn(); print(f'OK  {name}', flush=True)
    except Exception as e: print(f'ERR {name}: {type(e).__name__}: {str(e)[:90]}', flush=True)

df = cudf.DataFrame({'k':[1,2,1,3,2,1],'x':[1,2,3,4,5,6],'y':[10.,20.,30.,40.,50.,60.]})

# --- cudf-native compute + cudf-native readout via to_arrow (NO pandas) ---
def _gb_arrow():
    g = df.groupby('k').agg({'x':'sum'})
    xs = sorted(g.to_arrow().column('x').to_pylist())   # arrow, host, no pandas
    assert xs == [4,7,10]
run('groupby -> to_arrow (no pandas)', _gb_arrow)

def _gb_sortidx():
    g = df.groupby('k').agg({'x':'sum'}).sort_index()   # cudf-NATIVE sort_index (usa cupy?)
    assert g.to_arrow().column('x').to_pylist() == [10,7,4]
run('groupby + cudf.sort_index()', _gb_sortidx)

def _sortvals():
    s = df.sort_values('x', ascending=False)            # cudf-native sort
    assert s['x'].to_arrow().to_pylist()[0] == 6
run('sort_values (cudf-native)', _sortvals)

def _merge():
    r = cudf.DataFrame({'k':[1,2,3],'v':[100,200,300]})
    m = df.merge(r, on='k', how='left')
    assert len(m) == 6
run('merge (cudf-native)', _merge)

def _values():
    v = df['x'].values          # -> cupy array (GPU)
    assert int(v.sum()) == 21
run('.values (cupy array)', _values)

def _tocupy():
    import cupy
    c = df['x'].to_cupy()       # explicit cupy
    assert int(c.sum()) == 21
run('.to_cupy()', _tocupy)

def _scalar():
    assert int(df['x'].sum()) == 21 and int(df.groupby('k').agg({'x':'sum'})['x'].sum()) == 21
run('cudf scalars (no host convert)', _scalar)

print('DONE', flush=True)
