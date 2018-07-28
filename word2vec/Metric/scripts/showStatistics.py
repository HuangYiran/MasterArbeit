
from SegmentLevelData import SegmentLevelData
data = SegmentLevelData()
data.add_human_data("data/judgements.20150817.csv.gz")
print (data.extracted_pairs("de-en"))
print (data.extracted_pairs("en-de"))
