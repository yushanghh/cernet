import copy
import warnings
import pandas as pd
from alipy import ToolBox
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
train_data = pd.read_csv("Dataset/train_data.csv")
test_data = pd.read_csv("Dataset/test_data.csv")
train_label = train_data["FaultCause"]
train_data = train_data.drop(["FaultCause"], axis=1)
test_label = test_data["FaultCause"]
test_data = test_data.drop(["FaultCause"], axis=1)
all_data = pd.concat([train_data, test_data], axis=0)
all_label = pd.concat([train_label, test_label], axis=0)
all_data = all_data.values
all_label = all_label.values
all_resampled_data, all_resampled_label = SMOTE().fit_resample(all_data, all_label)

all_data = all_resampled_data
all_label = all_resampled_label

for index in range(0, len(all_label)):
    all_label[index] = all_label[index] - 1

alibox = ToolBox(X=all_data, y=all_label, query_type='AllLabels', saving_path='.')
alibox.split_AL(test_ratio=0.7, initial_label_rate=0.001, split_count=1)
model = alibox.get_default_model()
# model = AdaBoostClassifier(n_estimators=10)
# model = XGBClassifier(objective="reg:logistic")
# model = LogisticRegression()

# rft = SVC(kernel='linear')
# knn = KNeighborsClassifier(n_neighbors=7)

stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 500)


def main_loop(alibox, strategy, round):
    # Get the data split of one fold experiment
    train_idx, tet_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)

    # rft.fit(all_data[label_ind.index, :], all_label[label_ind.index])
    # pred = rft.predict(all_data[train_idx, :])
    # accuracy = alibox.calc_performance_metric(y_true=all_label[train_idx], y_pred=pred,
    #                                           performance_metric='accuracy_score')

    # Set initial performance point
    model.fit(X=all_data[label_ind.index, :], y=all_label[label_ind.index])
    pred = model.predict(all_data[train_idx, :])
    # knn.fit(all_data[label_ind.index, :], all_label[label_ind.index])
    # pred = knn.predict(all_data[train_idx, :])

    accuracy = alibox.calc_performance_metric(y_true=all_label[train_idx], y_pred=pred,
                                              performance_metric='accuracy_score')
    saver.set_initial_point(accuracy)
    # If the stopping criterion is simple, such as query 50 times. Use `for i in range(50):` is ok.
    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = strategy.select(label_index=label_ind, unlabel_index=unlab_ind, batch_size=1)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)
        model.fit(X=all_data[label_ind.index, :], y=all_label[label_ind.index])
        pred = model.predict(all_data[tet_idx, :])
        # rft.fit(all_data[label_ind.index, :], all_label[label_ind.index])
        # pred = rft.predict(all_data[tet_idx, :])
        # knn.fit(all_data[label_ind.index, :], all_label[label_ind.index])
        # pred = knn.predict(all_data[tet_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=all_label[tet_idx], y_pred=pred,
                                                  performance_metric='accuracy_score')
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        stopping_criterion.update_information(saver)
    stopping_criterion.reset()
    return saver


unc_entropy_result = []
unc_lc_result = []
unc_margin_result = []
qbc_result = []
eer_result = []
quire_result = []
density_result = []
bmdr_result = []
spal_result = []
lal_result = []
rnd_result = []

_I_have_installed_the_cvxpy = False

for round in range(1):

    train_idx, tet_idx, label_ind, unlab_ind = alibox.get_split(round)

    # Use pre-defined strategy
    # ['least_confident', 'margin', 'entropy', 'distance_to_boundary']
    unc_entropy = alibox.get_query_strategy(strategy_name="QueryInstanceUncertainty", measure='entropy')
    unc_lc = alibox.get_query_strategy(strategy_name="QueryInstanceUncertainty", measure='least_confident')
    unc_margin = alibox.get_query_strategy(strategy_name="QueryInstanceUncertainty", measure='margin')

    qbc = alibox.get_query_strategy(strategy_name="QueryInstanceQBC")
    # eer = alibox.get_query_strategy(strategy_name="QureyExpectedErrorReduction")
    rnd = alibox.get_query_strategy(strategy_name="QueryInstanceRandom")
    # quire = alibox.get_query_strategy(strategy_name="QueryInstanceQUIRE", train_idx=train_idx)
    # density = alibox.get_query_strategy(strategy_name="QueryInstanceGraphDensity", train_idx=train_idx)
    # lal = alibox.get_query_strategy(strategy_name="QueryInstanceLAL", cls_est=10, train_slt=False)
    # lal.download_data()
    # lal.train_selector_from_file(reg_est=30, reg_depth=5)

    unc_entropy_result.append(copy.deepcopy(main_loop(alibox, unc_entropy, round)))
    unc_lc_result.append(copy.deepcopy(main_loop(alibox, unc_lc, round)))
    unc_margin_result.append(copy.deepcopy(main_loop(alibox, unc_margin, round)))
    qbc_result.append(copy.deepcopy(main_loop(alibox, qbc, round)))
    # eer_result.append(copy.deepcopy(main_loop(alibox, eer, round)))
    rnd_result.append(copy.deepcopy(main_loop(alibox, rnd, round)))
    # quire_result.append(copy.deepcopy(main_loop(alibox, quire, round)))
    # density_result.append(copy.deepcopy(main_loop(alibox, density, round)))
    # lal_result.append(copy.deepcopy(main_loop(alibox, lal, round)))

    if _I_have_installed_the_cvxpy:
        bmdr = alibox.get_query_strategy(strategy_name="QueryInstanceBMDR", kernel='linear')
        spal = alibox.get_query_strategy(strategy_name="QueryInstanceSPAL", kernel='linear')

        bmdr_result.append(copy.deepcopy(main_loop(alibox, bmdr, round)))
        spal_result.append(copy.deepcopy(main_loop(alibox, spal, round)))

analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='Unc Entropy', method_results=unc_entropy_result)
analyser.add_method(method_name='Unc Least Confident', method_results=unc_lc_result)
analyser.add_method(method_name='Unc Margin', method_results=unc_margin_result)
analyser.add_method(method_name='QBC', method_results=qbc_result)
# analyser.add_method(method_name='EER', method_results=eer_result)
analyser.add_method(method_name='Random', method_results=rnd_result)
# analyser.add_method(method_name='QUIRE', method_results=quire_result)
# analyser.add_method(method_name='Density', method_results=density_result)
# analyser.add_method(method_name='LAL', method_results=lal_result)
if _I_have_installed_the_cvxpy:
    analyser.add_method(method_name='BMDR', method_results=bmdr_result)
    analyser.add_method(method_name='SPAL', method_results=spal_result)
print(analyser)
analyser.plot_learning_curves(title='Result', std_area=False)




