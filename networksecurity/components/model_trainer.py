import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model_comparison import ModelComparator

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
import mlflow.sklearn as msk
from urllib.parse import urlparse




class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self,best_model,classificationmetric):
        import dagshub
        dagshub.init(repo_owner='Divyanshb30', 
                 repo_name='Network-Security-System', 
                 mlflow=True)
       
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score

            

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.log_artifact("final_model/model.pkl", artifact_path="model")
           
    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,0.05,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)

        # ===========================
        # Upgrade 1: Model comparison
        # ===========================
        # NOTE: evaluate_models() mutates the `models` dict by fitting each estimator.
        # We take advantage of that here to register fitted models for comparison.
        comparator = ModelComparator(significance_level=0.05, random_state=42)

        previous_champion_name = "Champion_prev"
        previous_champion = None
        try:
            if os.path.exists("final_model/model.pkl"):
                previous_champion = load_object("final_model/model.pkl")
                comparator.register_model(previous_champion_name, previous_champion, version="previous")
        except Exception as e:
            logging.warning(f"Unable to load previous champion model. Proceeding without champion comparison. Error: {e}")

        registered_name_to_key = {}
        for model_key, fitted_model in models.items():
            registered_name = model_key.replace(" ", "_")
            registered_name_to_key[registered_name] = model_key
            comparator.register_model(registered_name, fitted_model, version="candidate")

        bootstrap_report = comparator.evaluate_all(x_test, y_test, n_bootstraps=100)
        candidate_names = [n for n in bootstrap_report.keys() if n != previous_champion_name]
        best_candidate_name = max(candidate_names, key=lambda n: bootstrap_report[n]["mean_f1"])

        comparison = None
        promote = True
        if previous_champion is not None:
            comparison = comparator.compare_models(previous_champion_name, best_candidate_name, x_test, y_test)
            promote = bool(comparison["significant"] and comparison["winner"] == best_candidate_name)

        # Small human-readable report (logs only; does not change pipeline behavior).
        report_lines = []
        report_lines.append("=== Model Comparison Report (Bootstrap F1 + McNemar) ===")
        for name in sorted(bootstrap_report.keys()):
            r = bootstrap_report[name]
            report_lines.append(
                f"- {name} (v={r['version']}): "
                f"mean_f1={r['mean_f1']:.4f}, std={r['std_f1']:.4f}, "
                f"95% CI=[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
            )
        if comparison is not None:
            c = comparison
            report_lines.append(
                f"McNemar: chi2={c['chi2_stat']:.4f}, p={c['p_value']:.6f}, "
                f"winner={c['winner']}, significant={c['significant']}"
            )
        promotion_text = "PROMOTE" if promote else "DON'T PROMOTE"
        report_lines.append(f"Promotion decision: {promotion_text} (alpha=0.05)")
        logging.info("\n".join(report_lines))

        # Select the best candidate from this training run by mean bootstrap F1.
        best_model_key = registered_name_to_key[best_candidate_name]
        best_model_name = best_model_key
        best_model = models[best_model_key]

        y_train_pred=best_model.predict(X_train)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        
        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)
        #model pusher
        save_object("final_model/model.pkl",best_model)
        self.track_mlflow(best_model,classification_test_metric)

        ## Track the experiements with mlflow
        self.track_mlflow(best_model,classification_train_metric)

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
     
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)