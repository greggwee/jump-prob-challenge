"""Download StatsBomb event files for all six tournaments (concurrent, cached)."""

from concurrent.futures import ThreadPoolExecutor

from jumpprob.ingest import statsbomb as sb

matches = sb.load_matches()
ids = matches["match_id"].tolist()
print(f"fetching events for {len(ids)} matches...")

done = 0
def fetch(mid):
    global done
    sb.fetch_events(mid)
    done += 1
    if done % 25 == 0:
        print(f"  {done}/{len(ids)}")

with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(fetch, ids))
print("done.")
