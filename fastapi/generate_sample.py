import pandas as pd
import os

# Dữ liệu mẫu
data = {
    "text": [
        "Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta sống và làm việc. Từ xe tự lái đến trợ lý ảo, AI giúp tăng hiệu suất và mở ra những cơ hội mới. Các công ty công nghệ lớn đang đầu tư hàng tỷ đô la vào nghiên cứu và phát triển AI.",
        "FastAPI là một web framework hiện đại, nhanh chóng (hiệu năng cao) để xây dựng các API với Python 3.8+ dựa trên các gợi ý kiểu chuẩn của Python. Các tính năng chính bao gồm: Nhanh, code nhanh, ít lỗi, trực quan, dễ dàng, ngắn gọn, mạnh mẽ và dựa trên các tiêu chuẩn.",
        "Biến đổi khí hậu đang là một thách thức toàn cầu đòi hỏi sự hợp tác của tất cả các quốc gia. Việc giảm lượng khí thải carbon và chuyển sang năng lượng tái tạo là những bước đi quan trọng để bảo vệ hành tinh của chúng ta cho các thế hệ tương lai."
    ],
    "reference": [
        "AI đang thay đổi cuộc sống và công việc, với sự đầu tư lớn từ các công ty công nghệ.",
        "FastAPI là framework Python hiện đại, hiệu năng cao, giúp xây dựng API nhanh chóng và ít lỗi.",
        "Biến đổi khí hậu cần sự hợp tác toàn cầu để giảm khí thải và bảo vệ hành tinh."
    ]
}

# Tạo DataFrame
df = pd.read_json(pd.io.json.dumps(data)) if not data else pd.DataFrame(data)

# Đảm bảo thư mục tồn tại
output_dir = os.path.dirname(os.path.abspath(__file__))
# Lưu ra ngoài thư mục fastapi một chút để dễ tìm, hoặc ngay trong root backend
output_path = os.path.join(output_dir, "..", "sample_dataset.xlsx")

# Xuất ra Excel
try:
    df.to_excel(output_path, index=False)
    print(f"Successfully created: {os.path.abspath(output_path)}")
except ImportError:
    # Fallback nếu thiếu openpyxl
    print("Missing openpyxl, installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "openpyxl"])
    df.to_excel(output_path, index=False)
    print(f"Successfully created: {os.path.abspath(output_path)}")
