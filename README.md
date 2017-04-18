Reset the home path to **YOUR_DATA_DIRECTORY** if necessary. The folder **archive/** contains old codes, and is not often used here.
Dependency: NLTK, TensorFlow, Keras ect.
1. Download the yelp challenge dataset
2. Run archive/data.py to export reviews of restaurants (stored in review_rest.json)
    There are two boolean values at the beginning.
        - (Optional) Set **do_checking=True** to just print out the categories for manual checking.
        - Set **do_export = True** to export the reviews required in the next step.
3. Run archive/preprocess/app_yelp/process_data.py to do tokenization and other preprocessing.
    Download the pretraining word vector model from [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip).
4. Run model/model_e2v_ntm.py to train a new model.
5. Run model/experiment.py to start the rating predicting experiment.
