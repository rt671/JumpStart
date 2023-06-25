import time, sys
from Base.Recommender_utils import seconds_to_biggest_unit

class Incremental_Training_Early_Stopping(object):
    """
    "Incremental" means that the model is updated at every epoch
    """

    def __init__(self):
        super(Incremental_Training_Early_Stopping, self).__init__()


    def get_early_stopping_final_epochs_dict(self):
        """
        This function returns a dictionary to be used as optimal parameters in the .fit() function
        """
        return {"epochs": self.epochs_best}



    def _run_epoch(self, num_epoch):
        """
        This function runs on a single epoch on the object we train.
        """
        raise NotImplementedError()


    def _prepare_model_for_validation(self):
        """
        This function is executed before the evaluation of the current model
        It should ensure the current object "self" can be passed to the evaluator object

        E.G. if the epoch is done via Cython or PyTorch, this function should get the new parameter values from the cython or pytorch objects into the self. pyhon object
        """
        raise NotImplementedError()


    def _update_best_model(self):
        """
        This function is called when the incremental model is found to have better validation score than the current best one
        """
        raise NotImplementedError()



    def _train_with_early_stopping(self, epochs_max, epochs_min = 0,
                                   validation_every_n = None, stop_on_validation = False,
                                   validation_metric = None, lower_validations_allowed = None, evaluator_object = None,
                                   algorithm_name = "Incremental_Training_Early_Stopping"):

        assert epochs_max > 0, "{}: Number of epochs_max must be > 0, passed was {}".format(algorithm_name, epochs_max)
        assert epochs_min >= 0, "{}: Number of epochs_min must be >= 0, passed was {}".format(algorithm_name, epochs_min)
        assert epochs_min <= epochs_max, "{}: epochs_min must be <= epochs_max, passed are epochs_min {}, epochs_max {}".format(algorithm_name, epochs_min, epochs_max)

        # Train for max number of epochs with no validation nor early stopping
        # OR Train for max number of epochs with validation but NOT early stopping
        # OR Train for max number of epochs with validation AND early stopping
        assert evaluator_object is None or\
               (evaluator_object is not None and not stop_on_validation and validation_every_n is not None and validation_metric is not None) or\
               (evaluator_object is not None and stop_on_validation and validation_every_n is not None and validation_metric is not None and lower_validations_allowed is not None),\
            "{}: Inconsistent parameters passed, please check the supported uses".format(algorithm_name)

        start_time = time.time()

        self.best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.epochs_best = 0

        epochs_current = 0

        while epochs_current < epochs_max and not convergence:

            self._run_epoch(epochs_current)

            # If no validation required, always keep the latest
            if evaluator_object is None:

                self.epochs_best = epochs_current

            # Determine whether a validaton step is required
            elif (epochs_current + 1) % validation_every_n == 0:

                print("{}: Validation begins...".format(algorithm_name))

                self._prepare_model_for_validation()

                # If the evaluator validation has multiple cutoffs, choose the first one
                results_run, results_run_string = evaluator_object.evaluateRecommender(self)
                results_run = results_run[list(results_run.keys())[0]]

                print("{}: {}".format(algorithm_name, results_run_string))

                # Update optimal model
                current_metric_value = results_run[validation_metric]

                if self.best_validation_metric is None or self.best_validation_metric < current_metric_value:

                    print("{}: New best model found! Updating.".format(algorithm_name))

                    self.best_validation_metric = current_metric_value

                    self._update_best_model()

                    self.epochs_best = epochs_current +1
                    lower_validatons_count = 0

                else:
                    lower_validatons_count += 1


                if stop_on_validation and lower_validatons_count >= lower_validations_allowed and epochs_current >= epochs_min:
                    convergence = True

                    elapsed_time = time.time() - start_time
                    new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                    # print("{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                    #     algorithm_name, epochs_current+1, validation_metric, self.epochs_best, self.best_validation_metric, new_time_value, new_time_unit))


            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            # print("{}: Epoch {} of {}. Elapsed time {:.2f} {}".format(
            #     algorithm_name, epochs_current+1, epochs_max, new_time_value, new_time_unit))

            epochs_current += 1

            sys.stdout.flush()
            sys.stderr.flush()

        # If no validation required, keep the latest
        if evaluator_object is None:

            self._prepare_model_for_validation()
            self._update_best_model()


        # Stop when max epochs reached and not early-stopping
        if not convergence:
            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            # if evaluator_object is not None:
            #     print("{}: Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
            #         algorithm_name, epochs_current, validation_metric, self.epochs_best, self.best_validation_metric, new_time_value, new_time_unit))
            # else:
            #     print("{}: Terminating at epoch {}. Elapsed time {:.2f} {}".format(
            #         algorithm_name, epochs_current, new_time_value, new_time_unit))

