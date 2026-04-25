import numpy as np

"""
        Lớp SVM dùng để cài đặt Soft-margin Support Vector Machine bằng NumPy
        Mục tiêu:
        - Tự xây dựng mô hình SVM, không dùng sklearn.svm.SVC 
        - Sử dụng hinge loss và regularization theo công thức Soft-margin SVM
        - Huấn luyện mô hình bằng phương pháp Stochastic Gradient Descent (SGD)

        Nhãn sử dụng:
        - normal = -1
        - pneumonia = 1
"""
class SVM:
    def __init__(self, C: float = 1.0, lr: float = 0.0001, epoch: int = 5):
        """
        Hàm khởi tạo các tham số ban đầu cho mô hình
        Tham số:
        - C: hệ số phạt của Soft-margin SVM
             C càng lớn thì mô hình càng phạt nặng các mẫu bị sai margin
        - lr: learning rate
        - epoch: số epoch, số lần mô hình đi qua toàn bộ tập train
        Thuộc tính:
        - self.w: vector trọng số của mô hình
        - self.b: bias của mô hình
        - self.losses: danh sách lưu loss sau mỗi epoch để theo dõi quá trình học
        """
        self.C = C
        self.lr = lr
        self.epoch = epoch
        self.w = None
        self.b = None
        self.losses = []

    def predict(self, X: np.ndarray):
        """
        Hàm tính điểm dự đoán thô của SVM
        Công thức:
            f(x) = Xw + b
        """
        return X @ self.w + self.b

    def predict_class(self, X: np.ndarray):
        """
        Hàm chuyển điểm dự đoán thô thành nhãn lớp cuối cùng
        - Nếu score >= 0 thì dự đoán là 1, tương ứng pneumonia
        - Nếu score < 0 thì dự đoán là -1, tương ứng normal
        Hàm flatten() được dùng để đưa kết quả từ dạng cột (N, 1) về vector 1 chiều (N,), thuận tiện cho việc tính precision, recall và F1-score
        """
        y_hat = self.predict(X)
        return np.where(y_hat >= 0, 1, -1).flatten()

    def loss_fn(self, X: np.ndarray, y: np.ndarray):
        """
        Hàm tính loss của Soft-margin SVM.
        Công thức:
            Loss = 1/2 ||w||^2 + C * mean(max(0, 1 - y * f(x)))
        Trong đó:
        - 1/2 ||w||^2 là thành phần giúp mô hình tránh overfitting và tạo margin rộng hơn
        - max(0, 1 - y * f(x)) là thành phần phạt các mẫu bị phân loại sai hoặc nằm trong margin
        - C điều chỉnh mức độ phạt của hinge loss
        Đầu vào:
        - X: dữ liệu ảnh sau khi tiền xử lý, shape (N, dim)
        - y: nhãn thật của dữ liệu, gồm -1 hoặc 1
        Đầu ra:
        - Giá trị loss tại thời điểm hiện tại
        """
        y = y.reshape(-1, 1)
        y_hat = self.predict(X)
        margins = 1 - y * y_hat
        return 0.5 * np.sum(self.w ** 2) + self.C * np.maximum(0, margins).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Hàm huấn luyện mô hình SVM bằng SGD
        Đầu vào:
        - X: ma trận dữ liệu train, shape (N, dim)
             N là số lượng ảnh
             dim là số đặc trưng của mỗi ảnh
             Với ảnh 128x128 đã flatten, dim = 16384
        - y: vector nhãn tương ứng, shape (N,)
        Quy trình huấn luyện:
        1. Khởi tạo trọng số w và bias b
        2. Với mỗi epoch, shuffle dữ liệu train
        3. Duyệt từng mẫu ảnh một
        4. Tính điểm dự đoán f(x)
        5. Kiểm tra điều kiện margin y_i * f(x_i) >= 1
        6. Tính gradient tương ứng
        7. Cập nhật w và b bằng SGD
        8. Sau mỗi epoch, tính loss và lưu vào self.losses
        """
        N, dim = X.shape
        self.w = np.zeros((dim, 1))
        self.b = 0.0
        y = y.reshape(-1, 1)

        for epoch in range(self.epoch):
            indices = np.random.permutation(N)
            X = X[indices]
            y = y[indices]

            for i in range(N):
                x_i = X[i].reshape(1, -1)
                y_i = y[i][0]

                y_hat_i = self.predict(x_i)[0, 0]

                if y_i * y_hat_i >= 1:
                    dW = self.w
                    db = 0
                else:
                    dW = self.w - self.C * y_i * x_i.T
                    db = -self.C * y_i

                self.w = self.w - self.lr * dW
                self.b = self.b - self.lr * db

            loss = self.loss_fn(X, y)
            self.losses.append(loss)

