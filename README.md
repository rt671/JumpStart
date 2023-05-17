Jumpstart: Tackling Cold Start Issue in Recommendations

Model for a hybrid recommender system, whose core model is a Content Based Filtering System, with Feature Weighting applied, such that they approximate the Collaborative Filtering Model

A significant challenge that many recommendation systems
face is the **cold start problem**. This problem arises when a system must make
recommendations for new users or items that have little to no historical data. In the case
of new users, the system has no information about their preferences, making it difficult
to make personalized recommendations. Similarly, for new items, there is no data on
how users have interacted with them in the past, making it challenging to predict how
a user might respond to them.

Observing the fact that there is a human tendency to prefer certain features over others, this fact was utilized in the model to modify the content based model through this feature weighting such that the resultant model approximates the collaborative model, thereby achieving the efficiency of collaborative filtering as well as 
the cold start resistance of content based filtering.

The System Model is described below:

![image](https://github.com/rt671/JumpStart/assets/82562103/49adb7bd-7bf6-42c2-a29e-ba174b955ee3)
