from pathlib import Path
from pprint import pprint

from data_tools import PremiumDataStore


store = PremiumDataStore(Path("data") / "Premium Variance Data.xlsx")

print("\nDATASET STATS\n")
pprint(store.dataset_stats)

print("\nCLIENT 001\n")
pprint(store.analyse_data(client="Client 001"))

print("\nTREATY 10001\n")
pprint(store.analyse_data(treaty=10001))

print("\nTERM / Q3\n")
pprint(store.analyse_data(portfolio="Term", quarter="Q3"))

print("\nAGE 45\n")
pprint(store.analyse_data(age_min=45, age_max=45))

print("\nCLIENT 001: Q1 VS Q4\n")
pprint(store.compare_quarters("Q1", "Q4", client="Client 001"))

print("\nWORST PERSISTENCY IN Q4\n")
pprint(
    store.rank_results(
        metric="LeftOver Persistency",
        group_by="Treaty",
        direction="ascending",
        top_n=10,
        quarter="Q4",
    )
)

print("\nINVALID PORTFOLIO\n")
pprint(store.analyse_data(portfolio="Universal Life"))
