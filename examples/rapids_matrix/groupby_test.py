import cudf
print('cudf', cudf.__version__, flush=True)
df = cudf.DataFrame({'k':[1,2,1,3,2,1],'x':[1,2,3,4,5,6],'y':[10.,20.,30.,40.,50.,60.]})
print('[gb] built df', flush=True)
g = df.groupby('k').agg({'x':'sum','y':'mean'}).sort_index()
print('[gb] groupby done', flush=True)
print(g.to_pandas(), flush=True)
assert int(g.loc[1,'x'])==10 and int(g.loc[2,'x'])==7 and int(g.loc[3,'x'])==4
assert abs(float(g.loc[1,'y'])-33.333)<0.01
print('[gb] GROUPBY OK', flush=True)
