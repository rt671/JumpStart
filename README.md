# Jumpstart
## Tackling Cold Start Issue in Recommendations

#### Machine Learning Model for a hybrid recommender system, whose core model is a Content Based Filtering System, with Feature Weighting applied to approximate the Collaborative Filtering Model

A significant challenge that many recommendation systems
face is the **cold start problem**. This problem arises when a system must make
recommendations for new users or items that have little to no historical data. In the case
of new users, the system has no information about their preferences, making it difficult
to make personalized recommendations. Similarly, for new items, there is no data on
how users have interacted with them in the past, making it challenging to predict how
a user might respond to them.

Observing the fact that there is a human tendency to prefer certain features over others, this fact was utilized in the model to modify the content based model through this feature weighting such that the resultant model approximates the collaborative model, thereby achieving the efficiency of collaborative filtering as well as 
the cold start resistance of content based filtering.

## System Model

There are two sources of data in our system which are the item profile and user profile,
and the eventual output of our system is the list of recommended items for each user.
The User profile contains the previous interactions of the user with various items, and
the Item profile is a matrix of items vs their features and each cell contains the relevance
of each feature for the item. 

![image](https://github.com/rt671/JumpStart/assets/82562103/49adb7bd-7bf6-42c2-a29e-ba174b955ee3)

The whole process is as follows:
1. **Data Preprocessing:** The user profile
and the item profiles are processed and converted
into suitable matrix formats for further processing, which are User Rating Matrix (URM) and Item Content Matrix (ICM).

2. **Collaborative Filtering:** The collaborative model learns over the training
URM and generates top K similar items for each item. This item-item
similarity matrix (IIM) is then fed for feature weighting as one of the inputs. CF
also handles the popularity bias.

3. **Feature Weighting:** It takes as input the training URM set, the ICM set, and
the Item similarity matrix which is obtained from the CF component. The item
features in the ICM set are modified and adjusted as per the IIM. This modified
ICM matrix along with URM is then passed to the core recommendation model
of our system, i.e. Content Based Model.

4. **Recommendation:** It takes the enhanced ICM and URM, and finds the recommended items for each user based on the
relevance of these features. These recommendations are provided to the users,
and users provide feedback in the form of ratings to those recommendations.
These feedbacks update the URM (and also add onto the user profile dataset)
and new recommendations are hence generated as per the URM. The ICM is
also updated when new movies or items are added or uploaded on the platform.

## Algorithm
The Model can be described in the following main steps:
1. **Building item similarity Matrices:** Building item-item similarity matrices for
both collaborative and content based filtering
2. **Identifying Similar Items:** Collecting similar feature item pairs also having
some collaborative similarity
3. **Machine Learning:** Initialize weights for each feature and loop over each item
pair identified in step 2, use common features between them and their feature
weights to generate a similarity value for the item pair. This is the calculated
similarity while the target similarity comes from the collaborative item-item
similarity matrix. The difference between the two is the loss. Apply gradient
descent to minimise this loss and learn the feature weights on the way. This optimization problem can be mathematically represented as:

![image](https://github.com/rt671/JumpStart/assets/82562103/9f1cde59-36a7-46e7-a21f-df3625891bd4)


4. **Modifying the Item Content Matrix:** Using the learned feature weights,
modify the item content matrix and generate the new hybrid item-item similarity
matrix
5. **Select the Suitable Candidates:** All the items similar to those items with which
the target user has positively interacted, are added to the candidate set

## Code Structure
- **application:** contains the application in Flask, contains the frontend of the application and app.py contains all functions that trigger the training, retraining and running the ML model.
- **model:** contains all the code to implement the ML model
    - **Data_Manager:** responsible for data fetching, preprocessing (cleaning, splitting) and creation of ICM and URM
    - **Base:** Base Recommender for Collaborative and Content Based Filtering Model. 
        - **Evaluator:** module for evaluation of the model
        - **Similarity:** module to compute Item similarity matrices 
        - **BaseRecommender:** abstract class for base recommender
        - **EarlyStopping:** implements Early Stopping, a regularization technique
    -  **Collaborative_Filtering:** Contains the code that outputs Item similarity matrix based on user interactions
    - **Feature_Weighting:** Contains the ML Model, Stochastic Gradient Descent, which learns the feature weights for content based IIM to approximate collaborative IIM.
    - matrices: Stores all the matrices developed in the process and required for retraining
    - **run_first_time.py:** **for the training of the model**
    - **retrain_model.py:** **For retraining of the model**, it is executed by the application, whenever the application is refreshed.
    - **gettopk.py:** after the model is learned, this function is called to return the final list of top k recommended items.

## Results
The Model's performance is comparable the many 'state-of-the-art' recommendation algorithms, especially in case of Cold Start scenarios.

Following is the comparison of NDCG values for various algorithms: 

![image](https://github.com/rt671/JumpStart/assets/82562103/fcc6256a-8730-4711-9a17-0066c75bb227)

Following is the User Interface showing the recommended movies to the user based on user's feedback (like/dislike) with the movies:

![image](https://github.com/rt671/JumpStart/assets/82562103/a6596f1c-23bc-4c99-829b-7a2ef9c34962)
