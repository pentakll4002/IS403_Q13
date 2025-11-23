import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    try:
        report = {}

        for name, model in models.items():
            # lấy grid params cho model hiện tại
            param_grid = params.get(name, {})

            # gridsearch để chọn tham số tốt nhất
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                scoring="r2",
                verbose=0
            )
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            # dự đoán
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # đánh giá
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            # lưu kết quả
            report[name] = {
                "best_params": gs.best_params_,
                "train_score": train_score,
                "test_score": test_score,
                "best_model": best_model
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
