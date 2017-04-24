#prepare the data set

import json
import os
home = os.environ["HOME"] +'/Data/yelp/'
# home = '/home/memray/Data/yelp/'

do_checking = True
do_export = True

if do_checking: #check category list
    # f = open("".join([home, "/data/yelp/business.json"]),"r")
    f = open("".join([home, "/yelp_academic_dataset_business.json"]),"r")
    categories_dict = {}
    interested_business_id = set()
    for line in f:
        obj = json.loads(line)
        business_id = obj["business_id"]
        categories = obj["categories"]

        if categories == None:
            continue

        keep_tags = ["Restaurants", "Food"]
        should_filter = True
        for tag in keep_tags:
            if tag in categories:
                should_filter = False
        if should_filter:
            continue

        interested_business_id.add(business_id)
        for category in categories:
            if category in categories_dict:
                categories_dict[category] += 1
            else:
                categories_dict[category] = 1

    categories_items = sorted(categories_dict.items(), key=lambda i:i[1], reverse=True)
    for id, (category, freq) in enumerate(categories_items):
        print('[%d]%s:%d' % (id, category, freq))
    print('Found %d categories' % len(categories_items))
    print('Found %d businesses' % len(interested_business_id))


if do_export: #only export reviews for Restaurants, output to "output/review_restaurant.json"
    f = open("".join([home, "/yelp_academic_dataset_business.json"]),"r")
    interested_business_id = set()

    for line in f:
        obj = json.loads(line)
        business_id = obj["business_id"]
        categories = obj["categories"]
        if categories == None:
            continue
        if "Restaurants" in categories or "Food" in categories:
            interested_business_id.add(business_id)

    f = open("".join([home, "yelp_academic_dataset_review.json"]),"r")
    fu = open("".join([home, "output/raw_review_restaurant.json"]),"w")
    nline = ""
    ncount = 0
    for line in f:
        obj = json.loads(line)
        business_id = obj["business_id"]

        # if this review is about a business of interest, append it to the output
        if business_id in interested_business_id:
            nline = "".join((nline,line))
        ncount += 1
        if ncount % 10000 == 0:
            fu.write(nline)
            print(ncount)
            nline = ""
    fu.write(nline)
    fu.close()

