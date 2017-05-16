Reset the home path to **YOUR_DATA_DIRECTORY** at the beginning of each code if necessary. The major codes  folder **archive/** contains old codes, and is not often used here.

Dependency: NLTK, TensorFlow, Keras ect.
1. Download the yelp challenge dataset, and I suggest to put it to /User/yourname/Data/yelp/ (Mac) or /home/yourname/Data/yelp/ (Linux)
2. Run archive/data.py to export reviews of restaurants (stored in HOME/output/raw_review_restaurant.json)
    There are two boolean values at the beginning.
        - (Optional) Set **do_checking=True** to just print out the categories for manual checking.
        - Set **do_export = True** to export the reviews required in the next step.
3. Run archive/preprocess/app_yelp/process_data.py to do tokenization and other preprocessing.
    Download the pretrained word vector model from [here](https://nlp.stanford.edu/projects/glove/).
4. Run entity/model_e2v_doc2vec.py, entity/model_e2v_ntm.py and entity/beta/model_e2v_attr.py to train a new model.
5. Run entity/model/user_rating.py to start the rating predicting experiment, and category_classification to start the classification and visualization.
