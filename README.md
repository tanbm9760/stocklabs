# 🚀 VN Stock AI Platform

Nền tảng phân tích và dự báo chứng khoán Việt Nam thông minh với giao diện web hiện đại.

## 🌟 Tính năng chính

### 📊 Stock Screener
- **Chấm điểm đa tiêu chí**: Phân tích kỹ thuật và cơ bản
- **Biểu đồ tương tác**: Hiển thị giá và khối lượng giao dịch
- **Lọc thông minh**: Theo ngành, vốn hóa, thanh khoản
- **Báo cáo AI**: Tích hợp OpenAI để tạo báo cáo chi tiết

### 🔮 LSTM Forecast
- **Mô hình Deep Learning**: Sử dụng LSTM để dự báo giá
- **Tùy chỉnh linh hoạt**: Điều chỉnh tham số huấn luyện
- **Đánh giá hiệu suất**: Metrics chi tiết (R², MAE, MAPE)
- **Lưu mô hình**: Tái sử dụng mô hình đã huấn luyện

## 🎨 Giao diện đã được tối ưu

### ✨ Cải tiến mới
- **Header gradient**: Thiết kế hiện đại với màu sắc gradient
- **Cards tương tác**: Hiệu ứng hover và shadow
- **Responsive design**: Tối ưu cho mọi kích thước màn hình
- **Loading states**: Progress bars và status indicators
- **Professional styling**: Màu sắc và typography nhất quán

### 🔧 Cấu trúc thư mục
```
vnStock/
├── Home.py                 # Trang chính
├── pages/
│   ├── 1_Screener.py      # Stock Screener
│   └── 2_LSTM_Forecast.py # LSTM Forecast
├── utils/
│   └── styling.py         # CSS utilities
├── styles/
│   └── common.css         # CSS chung
├── pick_best_by_symbols.py # Core logic
└── *.keras                # Saved models
```

## 🚀 Cài đặt và chạy

### Yêu cầu hệ thống
```bash
pip install streamlit pandas numpy plotly matplotlib
pip install tensorflow scikit-learn  # Cho LSTM
pip install vnstock openai requests  # Tùy chọn
```

### Chạy ứng dụng
```bash
# Từ thư mục vnStock
streamlit run Home.py
```

### Cấu hình
1. **API Keys**: Nhập OpenAI API key trong sidebar (tùy chọn)
2. **Nguồn dữ liệu**: Mặc định sử dụng TCBS, có thể thêm VCI, MSN
3. **Mô hình LSTM**: Tự động lưu và load từ file .keras

## 📱 Hướng dẫn sử dụng

### Stock Screener
1. **Quản lý Watchlist**: Tạo/sửa/xóa danh sách cổ phiếu
2. **Cấu hình tham số**: Điều chỉnh ngày, thanh khoản, nguồn dữ liệu
3. **Phân tích**: Nhấn "🚀 Bắt đầu phân tích"
4. **Xem kết quả**: Bảng xếp hạng, biểu đồ, lịch sử giá

### LSTM Forecast
1. **Chọn cổ phiếu**: Nhập mã cổ phiếu (VD: VIX, FPT)
2. **Cấu hình mô hình**: Window, epochs, batch size
3. **Huấn luyện**: Nhấn "🚀 Bắt đầu huấn luyện"
4. **Xem dự báo**: Biểu đồ performance và dự báo tương lai

## 🎯 Tính năng nâng cao

### Watchlist Management
- Lưu nhiều danh sách cổ phiếu
- Chia theo ngành, theme
- Tự động lọc mã hợp lệ

### AI Reports
- Tích hợp OpenAI GPT
- Báo cáo phân tích tự động
- Tùy chỉnh prompt và model

### Model Persistence
- Tự động lưu mô hình LSTM
- Load lại mô hình đã train
- Checkpoint tốt nhất

## 🛠️ Tùy chỉnh và mở rộng

### CSS Styling
- File `styles/common.css`: CSS variables và components
- File `utils/styling.py`: Python utilities cho styling
- Responsive design với grid system

### Performance
- Streamlit caching cho API calls
- Lazy loading cho charts
- Efficient data processing

### Security
- API key encryption
- Safe error handling
- Input validation

## 🔗 Tích hợp

### Nguồn dữ liệu
- **vnstock**: API chính thức VN
- **TCBS, VCI, MSN**: Nhiều nguồn backup
- **Real-time**: Cập nhật theo thời gian thực

### Machine Learning
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Metrics và preprocessing
- **Custom adapters**: Tích hợp vnstock

## 📊 Metrics và Đánh giá

### Stock Scoring
- **Value**: P/E, P/B ratios
- **Quality**: ROE, margins
- **Growth**: Revenue, earnings growth
- **Momentum**: Price trends
- **Liquidity**: Volume analysis

### LSTM Performance
- **R² Score**: Correlation accuracy
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Visual validation**: Charts comparison

## 🚀 Deployment

### Local Development
```bash
streamlit run Home.py --server.port 8501
```

### Production
- Streamlit Cloud
- Docker containerization
- Environment variables cho API keys

## 📝 License

Dự án mã nguồn mở cho cộng đồng đầu tư Việt Nam.

## 🤝 Đóng góp

Chào mừng các đóng góp:
- Bug reports
- Feature requests  
- Code improvements
- Documentation

---

**Phát triển bởi**: VN Stock AI Team  
**Cập nhật**: Tháng 9, 2025  
**Phiên bản**: 2.0 - Professional UI