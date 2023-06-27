import numpy as np

class Metrics_Object(object):
    # Abstract class used as superclass of all metrics requiring an object

    def __init__(self):
        pass

    def add_recommendations(self, recommended_items_ids):
        raise NotImplementedError()

    def get_metric_value(self):
        raise NotImplementedError()

    def merge_with_other(self, other_metric_object):
        raise NotImplementedError()

class MAP(Metrics_Object):
    # Mean Average Precision, defined as the mean of the AveragePrecision over all users

    def __init__(self):
        super(MAP, self).__init__()
        self.cumulative_AP = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant, pos_items):
        self.cumulative_AP += average_precision(is_relevant, pos_items)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_AP/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is MAP, "MAP: attempting to merge with a metric object of different type"

        self.cumulative_AP += other_metric_object.cumulative_AP
        self.n_users += other_metric_object.n_users


def roc_auc(is_relevant):

    ranks = np.arange(len(is_relevant))
    pos_ranks = ranks[is_relevant]
    neg_ranks = ranks[~is_relevant]
    auc_score = 0.0

    if len(neg_ranks) == 0:
        return 1.0

    if len(pos_ranks) > 0:
        for pos_pred in pos_ranks:
            auc_score += np.sum(pos_pred < neg_ranks, dtype=np.float32)
        auc_score /= (pos_ranks.shape[0] * neg_ranks.shape[0])

    assert 0 <= auc_score <= 1, auc_score
    return auc_score

def precision(is_relevant):

    if len(is_relevant) == 0:
        precision_score = 0.0
    else:
        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    assert 0 <= precision_score <= 1, precision_score
    return precision_score


def rmse(all_items_predicted_ratings, relevant_items, relevant_items_rating):
    # RMSE with test items
    relevant_items_error = (all_items_predicted_ratings[relevant_items]-relevant_items_rating)**2

    finite_prediction_mask = np.isfinite(relevant_items_error)

    if finite_prediction_mask.sum() == 0:
        rmse = np.nan

    else:
        relevant_items_error = relevant_items_error[finite_prediction_mask]

        squared_error = np.sum(relevant_items_error)
        mean_squared_error = squared_error/finite_prediction_mask.sum()
        rmse = np.sqrt(mean_squared_error)

    return rmse


def recall(is_relevant, pos_items):

    recall_score = np.sum(is_relevant, dtype=np.float32) / pos_items.shape[0]

    assert 0 <= recall_score <= 1, recall_score
    return recall_score

def average_precision(is_relevant, pos_items):

    if len(is_relevant) == 0:
        a_p = 0.0
    else:
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        a_p = np.sum(p_at_k) / np.min([pos_items.shape[0], is_relevant.shape[0]])

    assert 0 <= a_p <= 1, a_p
    return a_p


def ndcg(ranked_list, pos_items, relevance=None, at=None):

    if relevance is None:
        relevance = np.ones_like(pos_items)
    assert len(relevance) == pos_items.shape[0]

    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    # Creates array of length "at" with the relevance associated to the item in that position
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)

    # IDCG has all relevances to 1, up to the number of items in the test set
    ideal_dcg = dcg(np.sort(relevance)[::-1])

    # DCG uses the relevance of the recommended items
    rank_dcg = dcg(rank_scores)

    if rank_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg
    # assert 0 <= ndcg_ <= 1, (rank_dcg, ideal_dcg, ndcg_)
    return ndcg_


def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)
