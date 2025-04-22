import numpy as np  # Thư viện NumPy để xử lý mảng và ma trận số học.
import pandas as pd  # Thư viện Pandas để làm việc với dữ liệu dạng bảng (DataFrame).
from scipy.stats import f, chi2  # Hàm thống kê từ SciPy: f-distribution và chi-square distribution.
import matplotlib.pyplot as plt  # Thư viện Matplotlib để vẽ biểu đồ.
import seaborn as sns  # Thư viện Seaborn để tạo các biểu đồ đẹp và trực quan hóa dữ liệu.

# Load dataset
data = pd.read_csv("faults.csv")  # Đọc tập dữ liệu từ file CSV vào DataFrame.
print("Dataset shape:", data.shape)  # In ra kích thước của DataFrame (số hàng, số cột).
print("First few rows of the dataset:")  # In tiêu đề cho phần hiển thị dữ liệu.
print(data.head())  # Hiển thị 5 dòng đầu tiên của DataFrame để kiểm tra nội dung.

# Danh sách các biến được chọn để phân tích.
features = ['Steel_Plate_Thickness', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity']
sample_data = data[features]  # Lấy các cột tương ứng từ DataFrame gốc.
print("New Dataset shape:", sample_data.shape)  # In ra kích thước của DataFrame mới.
print(sample_data.head())  # Hiển thị 5 dòng đầu tiên của DataFrame mới.

# Vector trung bình tổng thể dựa trên giả định, hay còn được gọi là tiêu chuẩn của sản phẩm.
mu_0 = np.array([60, 1000, 100, 70, 150000])

# Calculate sample mean vector
x_bar = sample_data.mean().values  # Tính trung bình của từng cột trong DataFrame.
print("")
print(sample_data.mean())  # In ra vector trung bình mẫu.

S = sample_data.cov().values  # Tính ma trận hiệp phương sai của mẫu.
# Ma trận này mô tả mối quan hệ giữa các biến trong tập dữ liệu.
print(S)

# Number of samples and variables
n = sample_data.shape[0]  # Số lượng mẫu (số hàng trong DataFrame).
p = sample_data.shape[1]  # Số lượng biến (số cột trong DataFrame).

# Calculate Hotelling's T^2 statistic
T_squared = n * np.dot(np.dot((x_bar - mu_0).T, np.linalg.inv(S)), (x_bar - mu_0))
# Công thức tính Hotelling's T²:
# - (x_bar - mu_0): Hiệu giữa vector trung bình mẫu và vector trung bình tổng thể.
# - np.linalg.inv(S): Nghịch đảo của ma trận hiệp phương sai.
# - np.dot: Phép nhân ma trận và vector.

# Calculate critical value from F-distribution
alpha = 0.05  # Mức ý nghĩa (significance level) cho kiểm định.
F_critical = f.ppf(1 - alpha, p, n - p)   # Giá trị tới hạn từ phân phối F với bậc tự do p và n-p.
F_statistic = ((n - p) * T_squared) / (p * (n - 1))   # Tính giá trị thống kê F từ T².

# Output results
print(f"\nT^2 Statistic: {T_squared}")  # In ra giá trị thống kê T².
print(f"F Statistic: {F_statistic}")  # In ra giá trị thống kê F.
print(f"Critical F Value: {F_critical}")  # In ra giá trị tới hạn F.

# Hypothesis testing
# So sánh F_statistic với F_critical để đưa ra kết luận về giả thuyết H0.

if F_statistic > F_critical:
    print(f"\nBecause F statistic is greater than the critical value ({F_statistic} > {F_critical})")
    print("Reject H0 -> The sample mean is significantly different from the population mean.")
else:
    print(f"\nBecause F statistic is less than the critical value ({F_statistic} < {F_critical})")
    print("Fail to reject H0 -> The sample mean is not significantly different from the population mean.")

# Heatmap of correlation matrix
corr_matrix = sample_data.corr()  # Tính ma trận tương quan giữa các biến.
plt.figure(figsize=(8, 6))  # Thiết lập kích thước biểu đồ.
# Vẽ heatmap với chú thích (annot=True), màu sắc (cmap="coolwarm"), và định dạng số (fmt=".2f").
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")  # Thêm tiêu đề cho biểu đồ.
plt.tight_layout()  # Điều chỉnh layout để tránh chồng chéo.
plt.show()  # Hiển thị biểu đồ.

# Bar plot comparing sample mean and population mean for the first 4 variables
x_labels_4vars = features[:-1]  # Loại bỏ biến cuối cùng ('Sum_of_Luminosity').
x_indices_4vars = np.arange(len(x_labels_4vars))  # Tạo chỉ số cho các biến.

plt.figure(figsize=(10, 6))  # Thiết lập kích thước biểu đồ.
# Vẽ cột cho vector trung bình mẫu ($ \bar{x} $).
plt.bar(x_indices_4vars - 0.2, x_bar[:-1], width=0.4, label='Sample Mean ($\\bar{x}$)', color='blue')

# Vẽ cột cho vector trung bình tổng thể ($ \mu_0 $).
plt.bar(x_indices_4vars + 0.2, mu_0[:-1], width=0.4, label='Population Mean ($\\mu_0$)', color='orange')

plt.xticks(x_indices_4vars, x_labels_4vars, rotation=45)  # Đặt nhãn trục x và xoay 45 độ.
plt.ylabel("Values")  # Đặt nhãn trục y.
plt.ylim(0, 800)  # Giới hạn trục y để tập trung vào phạm vi quan trọng.
plt.title("Comparison of Sample Mean and Population Mean (First 4 Variables)")
# Thêm tiêu đề cho biểu đồ.
plt.legend()  # Thêm chú thích.
plt.tight_layout()  # Điều chỉnh layout.
plt.show()  # Hiển thị biểu đồ.

# Bar plot comparing sample mean and population mean for 'Sum_of_Luminosity'
x_labels_sum_luminosity = [features[-1]]  # Chỉ lấy biến cuối cùng ('Sum_of_Luminosity').
x_indices_sum_luminosity = np.arange(len(x_labels_sum_luminosity))  # Tạo chỉ số.

plt.figure(figsize=(8, 4))  # Thiết lập kích thước biểu đồ.
# Vẽ cột cho vector trung bình mẫu ($ \bar{x} $).
plt.bar(x_indices_sum_luminosity - 0.2, x_bar[-1], width=0.4, label='Sample Mean ($\\bar{x}$)', color='blue')
# Vẽ cột cho vector trung bình tổng thể ($ \mu_0 $).
plt.bar(x_indices_sum_luminosity + 0.2, mu_0[-1], width=0.4, label='Population Mean ($\\mu_0$)', color='orange')

plt.xticks(x_indices_sum_luminosity, x_labels_sum_luminosity, rotation=45)  # Đặt nhãn trục x.
plt.ylabel("Values")  # Đặt nhãn trục y.
plt.ylim(0, 200000)  # Giới hạn trục y.
# Thêm tiêu đề cho biểu đồ.
plt.title("Comparison of Sample Mean and Population Mean (Sum_of_Luminosity)")

plt.legend()  # Thêm chú thích.
plt.tight_layout()  # Điều chỉnh layout.
plt.show()  # Hiển thị biểu đồ.

