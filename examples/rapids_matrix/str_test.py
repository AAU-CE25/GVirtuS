import cudf
def run(name, fn):
    try: fn(); print(f'OK  {name}', flush=True)
    except Exception as e: print(f'ERR {name}: {type(e).__name__}: {str(e)[:90]}', flush=True)
df = cudf.DataFrame({'s':['abc','Def','gHi','JKL']})
print('[str] built string df', flush=True)
def _upper():
    assert df['s'].str.upper().to_arrow().to_pylist() == ['ABC','DEF','GHI','JKL']
run('str.upper', _upper)
def _lower():
    assert df['s'].str.lower().to_arrow().to_pylist() == ['abc','def','ghi','jkl']
run('str.lower', _lower)
def _contains():
    assert int(df['s'].str.contains('e').sum()) == 1
run('str.contains', _contains)
def _len():
    assert df['s'].str.len().to_arrow().to_pylist() == [3,3,3,3]
run('str.len', _len)
print('DONE', flush=True)
