import io
import pickle
import re
import ujson as json
import os

from archive.preprocess.util.text_process import TextProcess

# home = os.environ["HOME"]
home = os.environ["HOME"]

# path of input file, the raw review data
raw_data_path = "".join((home, "/Data/yelp/output/raw_review_restaurant.json"))
raw_dict_path = "".join((home, "/Data/yelp/output/raw_review_restaurant_dictioary.json"))
# path of pre-trained word embedding model
path_pretraining = "".join((home, "/Data/glove/glove.twitter.27B.200d.txt"))

filter_rest = True
business_id_set = set()
business_category_dict = {}

# path of output file, the review data after pre-processing
path_processed_output = "".join((home, "/Data/yelp/output/review_processed_rest_interestword_20170425_freq=100.txt"))
restaurant_of_interest_path = "".join((home, "/Data/yelp/output/restaurant_tag_of_interest.pkl"))
restaurant_review_pairs_path = "".join((home, "/Data/yelp/output/restaurant_review_pairs_freq=100.txt"))

'''
if True (obviously set to True), keep only the restaurants business
prods is a set containing the businesses, and prod_tag is a dict containing the categories of corresponding business
'''
if filter_rest:
    if os.path.exists(restaurant_of_interest_path):
        with open(restaurant_of_interest_path, 'rb') as f_:
            business_id_set, business_category_dict = pickle.load(f_)
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
        with open(restaurant_of_interest_path, 'wb') as f_:
            pickle.dump([business_id_set, business_category_dict], f_)
print('finish loading restaurants')

print('#(business)=%d' % len(business_id_set))

'''
process reviews and export to the following format, each line consists of three parts:
business_id\ttag1 tag2 ... tagn\tprocessed review text

Also export the business_id\ttext pairs
'''
print('initializing word vector')

# build dictionary and count terms
dictionary = {}
if os.path.exists(raw_dict_path):
    print('loading dictionary')
    with open(raw_dict_path, 'r') as f_dict:
        dictionary = json.load(f_dict)
else:
    print('building dictionary')
    f_data = open(raw_data_path, "r")
    for l_id, line in enumerate(f_data):
        obj = json.loads(line)
        business_id = obj["business_id"]

        # only do process for interesting businesses
        if business_id in business_id_set:
            words = re.split("\W+", obj["text"].lower())
            for w in words:
                dictionary[w] = dictionary.get(w, 0) + 1

        if l_id % 10000 == 0:
            print(l_id)

    with open(raw_dict_path, 'w') as f_dict:
        json.dump(dictionary, f_dict)
    with open(raw_dict_path+'.txt', 'w') as f_dict:
        freq_list = sorted(dictionary.items(), key=lambda t:t[1], reverse=True)
        for w,freq in freq_list:
            f_dict.write('%s\t%d\n' % (w,freq))

textProcess = TextProcess(dictionary)
textProcess.initiliaze(path_pretraining)


f_processed = open(path_processed_output, "w", buffering=io.DEFAULT_BUFFER_SIZE * 8)
import time

start = time.time()

current_business_id = ''
current_business_tag = ''
current_business_reviews = ''
business_review_pair_writer = open(restaurant_review_pairs_path, 'w', buffering=io.DEFAULT_BUFFER_SIZE * 8)

f_data = open(raw_data_path, "r")
for n_count, line in enumerate(f_data):
    obj = json.loads(line)
    business_id = obj["business_id"]

    # only do process for interesting businesses
    if business_id in business_id_set:
        text = textProcess.process(obj["text"])

        # user_id = str(obj["user_id"])
        # stars = str(obj["stars"])
        tags = " ".join(business_category_dict[business_id])
        f_processed.write(business_id +"\t"+ tags +"\t"+ text + "\n")

        if business_id != current_business_id:
            business_review_pair_writer.write(current_business_id+'\t'+current_business_tag+'\t'+current_business_reviews+'\n')
            current_business_id = business_id
            current_business_reviews = text
            current_business_tag = tags
        else:
            current_business_reviews += text

    n_count += 1
    if n_count % 50000 == 0:
        print(n_count)
        print(time.time() - start)

# flush the last output
business_review_pair_writer.write(current_business_id + '\t' + current_business_reviews)
# TextProcess.generateStemPair()
f_processed.close()
business_review_pair_writer.close()