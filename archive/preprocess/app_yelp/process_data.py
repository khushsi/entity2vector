import json
import os

from archive.preprocess.util.text_process import TextProcess

# home = os.environ["HOME"]
home = os.environ["HOME"]

# path of input file, the raw review data
path = "".join((home, "/Data/yelp/output/raw_review_restaurant.json"))
# path of output file, the review data after pre-processing
path_processed_output = "".join((home, "/Data/yelp/output/review_processed_rest_interestword_20170418.txt"))
# path of pre-trained word embedding model
path_pretraining = "".join((home, "/Data/glove/glove.twitter.27B.200d.txt"))

filter_rest = True
prods = set()
prod_tag = {}

'''
if True (obviously set to True), keep only the restaurants business
prods is a set containing the businesses, and prod_tag is a dict containing the categories of corresponding business
'''
if filter_rest:
    path_prod = "".join((home, "/Data/yelp/yelp_academic_dataset_business.json"))
    f_prod = open(path_prod, "r")
    for line in f_prod:
        obj = json.loads(line)
        business_id = str(obj["business_id"])
        categories = obj["categories"]
        if categories!=None and ("Restaurants" in categories or "Food" in categories):
            prods.add(business_id)
            if not business_id in prod_tag:
                prod_tag[business_id] = set()
            for category in categories:
                if "Restaurants" != category and "Food" != category:
                    prod_tag[business_id].add(category.replace(" ",""))



TextProcess.initiliaze(path_pretraining)
f = open(path, "r")
f_processed = open(path_processed_output, "w")
batch = ""
n_count = 0

for line in f:
    obj = json.loads(line)
    text = TextProcess.process(obj["text"])
    user_id = str(obj["user_id"])
    business_id = str(obj["business_id"])
    stars = str(obj["stars"])

    if not filter_rest or business_id in prods:
        tags = " ".join(prod_tag[business_id])
        line_processed = "\t".join((business_id,tags, text))
        batch = "\n".join((batch, line_processed))
        n_count += 1
        if n_count % 10000 == 0:
            print(n_count)
            f_processed.write(batch)
            f_processed.write("\n")
            batch = ""
f_processed.write(batch)
f_processed.write("\n")
TextProcess.generateStemPair()

