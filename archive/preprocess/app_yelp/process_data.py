import io
import pickle
import ujson as json
import os

from archive.preprocess.util.text_process import TextProcess

# home = os.environ["HOME"]
home = os.environ["HOME"]

# path of input file, the raw review data
path = "".join((home, "/Data/yelp/output/raw_review_restaurant.json"))
# path of output file, the review data after pre-processing
path_processed_output = "".join((home, "/Data/yelp/output/review_processed_rest_interestword_20170425.txt"))
# path of pre-trained word embedding model
path_pretraining = "".join((home, "/Data/glove/glove.twitter.27B.200d.txt"))

filter_rest = True
business_id_set = set()
business_category_dict = {}

restaurant_of_interest_path = "".join((home, "/Data/yelp/output/restaurant_tag_of_interest.pkl"))
restaurant_review_pairs_path = "".join((home, "/Data/yelp/output/restaurant_review_pairs.txt"))

'''
if True (obviously set to True), keep only the restaurants business
prods is a set containing the businesses, and prod_tag is a dict containing the categories of corresponding business
'''
if filter_rest:
    if os.path.exists(restaurant_of_interest_path):
        with open(restaurant_of_interest_path, 'rb') as f:
            business_id_set, business_category_dict = pickle.load(f)
    else:
        path_prod = "".join((home, "/Data/yelp/yelp_academic_dataset_business.json"))
        f_prod = open(path_prod, "r")
        for line in f_prod:
            obj = json.loads(line)
            business_id = str(obj["business_id"])
            categories = obj["categories"]
            if categories!=None and ("Restaurants" in categories or "Food" in categories):
                business_id_set.add(business_id)
                if not business_id in business_category_dict:
                    business_category_dict[business_id] = set()
                for category in categories:
                    # keep the categories that are not "restaurants" or "food"
                    if "Restaurants" != category and "Food" != category:
                        business_category_dict[business_id].add(category.replace(" ", ""))
        with open(restaurant_of_interest_path, 'wb') as f:
            pickle.dump([business_id_set, business_category_dict], f)
print('finish loading restaurants')

'''
process reviews and export to the following format, each line consists of three parts:
business_id\ttag1 tag2 ... tagn\tprocessed review text

Also export the business_id\ttext pairs
'''
print('initializing word vector')

textProcess = TextProcess()
TextProcess.initiliaze(path_pretraining)
f = open(path, "r")
f_processed = open(path_processed_output, "w", io.DEFAULT_BUFFER_SIZE * 8)
n_count = 0
import time

start = time.time()

business_text_dict = {}
for b_id in business_id_set:
    business_text_dict[b_id] = ''

for line in f:
    obj = json.loads(line)
    business_id = obj["business_id"]

    # only do process for interesting businesses
    if business_id in business_id_set:
        text = textProcess.process(obj["text"])
        # business_text_dict[b_id] += (text + ' ')

        # user_id = str(obj["user_id"])
        # stars = str(obj["stars"])
        tags = " ".join(business_category_dict[business_id])
        line_processed = business_id +"\t"+ tags #+"\t"+ text
        f_processed.write(line_processed + "\n")

    n_count += 1
    if n_count % 10000 == 0:
        print(n_count)
        print(time.time() - start)

# TextProcess.generateStemPair()
f_processed.close()

print('Exporting business-review pairs')
with open(restaurant_review_pairs_path, 'w') as f:
    for idx, b_id in enumerate(business_id_set):
        if idx % 1000 == 0:
            print(idx)
        f.write(b_id+'\t'+business_text_dict[b_id]+'\n')

print('Exporting business-review pairs: Done!')