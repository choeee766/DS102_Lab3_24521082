import numpy as np
import matplotlib.pyplot as plt
from load_dataset import collect, Scaler
from model import SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def metric(y_true, y_pred, name):
    """
    Tính các độ đo đánh giá mô hình
    
    TP = phát hiện đúng người bị viêm phổi (True Positive)
    FP = báo nhầm người bình thường thành viêm phổi (False Positive)
    FN = bỏ sót người bị viêm phổi (False Negative)
    TN = nhận diện đúng người bình thường (True Negative)

    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    TN = np.sum((y_true == -1) & (y_pred == -1))

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"TP: {TP} | FP: {FP} | FN: {FN} | TN: {TN}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def from_scratch(X_train, y_train, X_test, y_test):
    """
    Bài 1: Gọi mô hình SVM tự cài đặt bằng NumPy và train bằng SGD
    """
    model = SVM(C=1.0, lr=0.0001, epoch=5)
    print("\nBÀI 1: SVM TỰ CÀI ĐẶT BẰNG NUMPY")
    model.fit(X_train, y_train)
    y_pred = model.predict_class(X_test)

    return metric(y_test, y_pred, "Metric from Scratch")

def library(X_train, y_train, X_test, y_test):
    """
    Bài 2: Gọi mô hình SVM từ thư viện sklearn
    """
    model = SVC(kernel="linear", C=1.0)
    print("\nBÀI 2: SVM SỬ DỤNG THƯ VIỆN SKLEARN")
    print("Đang train SVC(kernel='linear')...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return metric(y_test, y_pred, "Metric from Standard Library")

def plot_comparison(result_1, result_2):
    """
    Vẽ biểu đồ cột so sánh kết quả bài 1 và bài 2.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

    scratch_scores = [
        result_1["accuracy"],
        result_1["precision"],
        result_1["recall"],
        result_1["f1"]
    ]

    library_scores = [
        result_2["accuracy"],
        result_2["precision"],
        result_2["recall"],
        result_2["f1"]
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, scratch_scores, width, label="SVM tự cài đặt")
    bars2 = plt.bar(x + width / 2, library_scores, width, label="SVM dùng thư viện")

    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Giá trị đánh giá")
    plt.title("So sánh kết quả giữa SVM tự cài đặt và SVM dùng thư viện")
    plt.legend()

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig("Bang_so_sanh.png", dpi=300)
    plt.show()


def main():
    """
    - Đọc dữ liệu ảnh.
    - Chuẩn hóa dữ liệu.
    - Chạy SVM tự cài đặt.
    - Chạy SVM thư viện.
    - So sánh kết quả bằng biểu đồ cột.
    """
    X_train, y_train = collect("train")
    X_test, y_test = collect("test")

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    result_1 = from_scratch(X_train, y_train, X_test, y_test)
    result_2 = library(X_train, y_train, X_test, y_test)

    plot_comparison(result_1, result_2)

if __name__ == "__main__":
    main()