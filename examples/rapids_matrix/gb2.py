import cudf
def s(m): print(f'[gb] {m}', flush=True)
df = cudf.DataFrame({'k':[1,2,1,3,2,1],'x':[1,2,3,4,5,6]})
s('built df')
g = df.groupby('k').agg({'x':'sum'})
s('agg computed')
h = g.to_pandas()
s('to_pandas done')
h = h.sort_index()
print(h, flush=True)
assert int(h.loc[1,'x'])==10 and int(h.loc[2,'x'])==7 and int(h.loc[3,'x'])==4
s('AGG OK (groupby sum correcto)')
