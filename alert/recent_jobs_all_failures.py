def recent_jobs_all_failures(df):
  """All jobs failed on the last day that the user ran."""
  cols = ["netid", "cluster", "state", "start", "end"]
  f = df[cols][df.start != "Unknown"].copy()
  # next line deals with RUNNING jobs
  f["end"] = f["end"].str.replace("Unknown", str(round(time.time())), regex=False)
  f["end"] = f["end"].astype("int64")
  def day_since(x):
    dt = datetime.fromtimestamp(x)
    return (datetime(dt.year, dt.month, dt.day) - datetime(1970, 1, 1)).days
  f["day-since-epoch"] = f["end"].apply(day_since)
  d = {"netid":np.size, "state":lambda series: sum([s == "FAILED" for s in series])}
  f = f.groupby(["netid", "cluster", "day-since-epoch"]).agg(d).rename(columns={"netid":"jobs", "state":"num-failed"}).reset_index()
  f = f.groupby(["netid", "cluster"]).apply(lambda d: d.iloc[d["day-since-epoch"].argmax()])
  f = f[(f["num-failed"] == f["jobs"]) & (f["num-failed"] > 3)]
  f["dossier"]  = f.netid.apply(lambda x: dossier.ldap_plus([x])[1])
  f["position"] = f.dossier.apply(lambda x: x[3])
  f["dept"]     = f.dossier.apply(lambda x: x[1])
  filters = ~f["position"].isin(["G3", "G4", "G5", "G6", "G7", "G8"])
  return f[["netid", "position", "dept", "cluster", "jobs", "num-failed"]][filters]
