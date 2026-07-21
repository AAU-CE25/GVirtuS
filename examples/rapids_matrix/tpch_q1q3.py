import time, numpy as np, cudf
print("cudf", cudf.__version__, flush=True)
rng = np.random.default_rng(42)
NL, NO, NC = 600000, 150000, 15000   # lineitem / orders / customer rows

t0=time.time()
cust = cudf.DataFrame({
    "c_custkey": np.arange(NC, dtype="int64"),
    "c_mktsegment": rng.choice(["BUILDING","AUTOMOBILE","MACHINERY","HOUSEHOLD","FURNITURE"], NC),
})
orders = cudf.DataFrame({
    "o_orderkey": np.arange(NO, dtype="int64"),
    "o_custkey": rng.integers(0, NC, NO, dtype="int64"),
    "o_orderdate": rng.integers(8000, 9500, NO, dtype="int32"),
    "o_shippriority": rng.integers(0, 5, NO, dtype="int32"),
})
lineitem = cudf.DataFrame({
    "l_orderkey": rng.integers(0, NO, NL, dtype="int64"),
    "l_quantity": rng.integers(1, 51, NL).astype("float64"),
    "l_extendedprice": (rng.random(NL)*100000).astype("float64"),
    "l_discount": (rng.random(NL)*0.1).astype("float64"),
    "l_tax": (rng.random(NL)*0.08).astype("float64"),
    "l_returnflag": rng.choice(["A","N","R"], NL),
    "l_linestatus": rng.choice(["O","F"], NL),
    "l_shipdate": rng.integers(8000, 9500, NL, dtype="int32"),
})
print(f"data built ({NL} lineitem rows) in {(time.time()-t0)*1000:.0f} ms", flush=True)

def q1():
    df = lineitem[lineitem.l_shipdate <= 9400]
    df = df.assign(disc_price=df.l_extendedprice*(1-df.l_discount))
    df = df.assign(charge=df.disc_price*(1+df.l_tax))
    return df.groupby(["l_returnflag","l_linestatus"]).agg({
        "l_quantity":"sum","l_extendedprice":"sum","disc_price":"sum",
        "charge":"sum","l_discount":"mean","l_orderkey":"count"}).sort_index()

def q3():
    c = cust[cust.c_mktsegment=="BUILDING"]
    o = orders[orders.o_orderdate < 9000]
    l = lineitem[lineitem.l_shipdate > 9000]
    co = c.merge(o, left_on="c_custkey", right_on="o_custkey")
    col = co.merge(l, left_on="o_orderkey", right_on="l_orderkey")
    col = col.assign(revenue=col.l_extendedprice*(1-col.l_discount))
    r = col.groupby(["l_orderkey","o_orderdate","o_shippriority"]).agg({"revenue":"sum"})
    return r.reset_index().sort_values("revenue", ascending=False).head(10)

for name, fn in [("Q1",q1),("Q3",q3)]:
    t=time.time()
    try:
        res = fn(); n=len(res); dt=(time.time()-t)*1000
        print(f"OK  {name}: {n} result rows in {dt:.0f} ms", flush=True)
        print(res.to_pandas().to_string()[:700], flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"ERR {name}: {type(e).__name__}: {str(e)[:140]}", flush=True)
print("DONE", flush=True)
